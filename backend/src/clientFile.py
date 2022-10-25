import io
import asyncio
import json
import time
from aiohttp import web
import logging

logger = logging.getLogger(__name__)


class clientFile(io.RawIOBase):
    def __init__(
        self,
        ws: web.WebSocketResponse,
        metadata: dict,
        max_chunk_size: int = 4 * 1024 * 1024 - 8,
        chunk_timeout: float = 5,
        max_pending_requests: int = 10,
    ):
        """clientFile initializer. This class abstracts client file in the browser
        as a python file-like object on the server. This class communicates with client
        side java-script over WebSocket handle `ws`. Client drags the file on the website,
        and we mount the file to this class. Requests to read and seek are translated to
        websocket requests for file slice, then over the same webSocket the bytes are
        recieved here and the read() returns these bytes.

        There are two types of exchanged messages (ws needs to adhere to the following
        specification):
        server sends range requests to client as string that is json serialized dict:
            { "msg": "get", "S": offset, "E": offset + bytes_to_pull_from_client,}
        it is expected that client will respond with binary websocket message containing
            file slice of bytes from "S" to "E", preceded by 8-byte integer, big-endian
            ordered (byteorder='big'), with value equal to the index of starting byte "S".

        Args:
            ws (web.WebSocketResponse): WebSocket handle to communicate with client
            metadata (dict): metadata about the file from client's javascript,
                proper "size" info is required for this class instances to work
            max_chunk_size (int, optional): max length of WebSocket messages to use.
                Defaults to 4*1024*1024-8.
            chunk_timeout (float, optional): max time to wait for requested file slice.
                Defaults to 5.
            max_pending_requests (int, optional): maximum number of requests sent to
                client before backing off. Defaults to 10.
        """
        # initially based on
        # https://github.com/mariobuikhuizen/ipyvuetify/blob/master/ipyvuetify/extra/file_input.py
        # (MIT License)
        self.ws = ws
        self.metadata = metadata
        self.fake_filename = metadata.get("name", "unknown")
        self.__name__ = self.fake_filename
        self.size = metadata["size"]
        self.offset = 0
        self.max_chunk_size = max_chunk_size
        self.data_requests = {
            # offset: {"buffer": buff, "start":timestamp_at_request_init}
        }
        self.timeout = chunk_timeout
        self.max_pending_requests = max_pending_requests

    def seekable(self) -> bool:
        """seekable property.

        Returns:
            bool: True
        """
        return True

    def readable(self) -> bool:
        """readable property.

        Returns:
            bool: True
        """
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET):
        """set reading head of the file object to given offset.

        Args:
            offset (int): offset to set.
            whence (int, optional): one of [SEEK_SET, SEEK_CUR, SEEK_END].
                Mode of seeking. Defaults to io.SEEK_SET.

        Raises:
            ValueError: in case of invalid whence
        """
        if whence == io.SEEK_SET:
            self.offset = offset
        elif whence == io.SEEK_CUR:
            self.offset = self.offset + offset
        elif whence == io.SEEK_END:
            self.offset = self.size + offset
        else:
            raise ValueError(f"whence {whence} invalid")

    def tell(self) -> int:
        """tells the offset of file object's reading head

        Returns:
            int: current offset
        """
        return self.offset

    def _cleanup_timedout_requests(self):
        """callback that cleans internal state such that requests that timed-out are no longer awaited for"""
        request_offsets = list(self.data_requests.keys())
        curr_time = time.time()
        for o in request_offsets:
            if self.data_requests[o]["start"] + self.timeout < curr_time:
                logger.warning(
                    f"popping request (for offset {o}) from request stack. req. age: {curr_time-self.data_requests[o]['start']:.3f}s"
                )
                self.data_requests.pop(o)

    async def readinto(self, buffer: bytearray) -> int:
        """reads file starting from current offset into buffer.

        Args:
            buffer (bytearray): array to which data is written.

        Returns:
            int: number of bytes read.
        """
        offset = self.offset

        if offset > self.size:
            return 0
        bytes_read = 0
        mem = memoryview(buffer)

        remaining_bytes_to_EOF = max(0, self.size - self.offset)
        size = min(len(buffer), remaining_bytes_to_EOF)
        sleep_interval = 0.01
        max_iterations = self.timeout / sleep_interval
        current_task = asyncio.current_task()
        while bytes_read < size:
            bytes_to_pull_from_client = min(self.max_chunk_size, size - bytes_read)
            while self.max_pending_requests < len(self.data_requests):
                logger.debug(f"too many requests pending, sleeping")
                self._cleanup_timedout_requests()  # should not be needed, was used to fix a problem that is now solved (I hope)
                await asyncio.sleep(sleep_interval)
            while offset in self.data_requests:
                logger.debug(
                    f"request starting at offset {offset} alredy queued, sleeping"
                )
                await asyncio.sleep(sleep_interval)
            # while bytes_to_pull_from_client in self.data_requests:
            #     logger.debug(f"request of {bytes_to_pull_from_client} bytes already in queue")
            #     bytes_to_pull_from_client -= 1
            logger.debug(
                f"asking client for {bytes_to_pull_from_client} bytes of data, offset:{self.offset}"
            )
            # self.data_requests[offset] = None
            self.data_requests[offset] = {
                "buffer": None,
                # "bytes to pull": bytes_to_pull_from_client,
                "start": time.time(),
            }

            def callback_in_case_of_cancellation(fut):
                self.data_requests.pop(offset)

            current_task.add_done_callback(
                callback_in_case_of_cancellation
            )  # protect requests queue from cancellation which could lead to handle leaks
            await self.ws.send_str(
                json.dumps(
                    {
                        "msg": "get",
                        "S": offset,
                        "E": offset + bytes_to_pull_from_client,
                    }
                )
            )
            iterations = 0
            while self.data_requests[offset]["buffer"] is None:
                if iterations > max_iterations:
                    logger.warning(
                        f"timeout! offset: {offset}, bytes_to_pull_from_client: {bytes_to_pull_from_client}"
                    )
                    return 0
                    # raise Exception('Timeout')
                await asyncio.sleep(sleep_interval)
                iterations += 1
            current_task.remove_done_callback(
                callback_in_case_of_cancellation
            )  # stop protecting queue from cancellation - no more awaits from here
            # awaited_data_buffer = self.data_requests.pop(offset)
            result = self.data_requests.pop(offset)
            awaited_data_buffer = result.pop("buffer")
            bytes_pulled_from_client = len(awaited_data_buffer)
            if bytes_pulled_from_client != bytes_to_pull_from_client:
                logger.warning(
                    "WARN: bytes_pulled_from_client != bytes_to_pull_from_client; corrupted comminication detected"
                )
            logger.debug(
                f"writing to buffer, off:{offset}, (self.off:{self.offset}), result: {result} bytes_to_pull_from_client: {bytes_pulled_from_client}"
            )
            safe__bytes_pulled_from_client = min(
                self.max_chunk_size, bytes_pulled_from_client
            )
            safe__bytes_pulled_from_client = min(
                safe__bytes_pulled_from_client, len(mem) - bytes_read
            )
            mem[
                bytes_read : bytes_read + safe__bytes_pulled_from_client
            ] = awaited_data_buffer[:safe__bytes_pulled_from_client]
            # mem[bytes_read:bytes_read + bytes_pulled_from_client] = awaited_data_buffer
            bytes_read += safe__bytes_pulled_from_client
            offset += safe__bytes_pulled_from_client
        self.offset = offset
        logger.debug(f"readinto(): done, bytes_read: {bytes_read}")
        return bytes_read

    async def readall(self) -> bytearray:
        """reads all remaining data from current offset to EOF.

        Returns:
            bytearray: data read.
        """
        return await self.read(self.size - self.offset)

    async def read(self, size: int) -> bytearray:
        """reads up to `size` bytes.

        Args:
            size (int): bytes to read.

        Returns:
            bytearray: array of bytes of length at most `size`.
        """
        logger.debug(f"requested to read {size} bytes")
        buf = bytearray(size)
        read = await self.readinto(buf)
        return buf[:read]

    def receive_data(self, msg: bytearray):
        """callback that consumes and interprets binary data from client.

        Args:
            msg (bytearray): binary data recieved from client.
        """
        if len(msg) <= 8:
            logger.warn(f"got message of size <= 8. len(msg): {len(msg)} bytes")
            return
        offset = int.from_bytes(msg[:8], byteorder="big")
        msg = msg[8:]
        # msg_len = len(msg)
        if offset not in self.data_requests:
            logger.warn(
                f"offset not in self.data_requests. offset: {offset} msg_len {len(msg)} bytes"
            )
            return

        self.data_requests[offset]["buffer"] = msg
