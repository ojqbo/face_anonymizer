import asyncio
import json

import logging
from typing import Awaitable

logger = logging.getLogger(__name__)


class workerConnection:
    def __init__(self, url, host: str = "localhost", port: int = 13263):
        """connection to the compute worker.
        `await workerConnection_instance.init()` is necessary to establish communication.

        Args:
            url (str): URL where the client uploaded file is available
            host (str, optional): host where the compute worker is located. Defaults to "localhost".
            port (int, optional): port on which the compute worker listens. Defaults to 13263.
        """
        assert isinstance(url, str)
        self.url = url
        self.host = host
        self.port = port

    async def init(self) -> bool:
        """initiates communication with worker.
        Will save response worker welcome message in parsed_metadata property.

        Returns:
            bool: True if connection openned successfully, False otherwise
        """
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host,
                self.port,
                limit=1024 * 1024,  # None,
            )
        except ConnectionRefusedError as err:
            logger.error(f"ConnectionRefusedError: {err}")
            self.ok = False
            return self.ok

        logger.debug(f"worker connected {self.host}:{self.port}")
        msg = json.dumps({"src": self.url}) + "\n"
        self.writer.write(msg.encode())
        await self.writer.drain()

        welcome_msg = await self.reader.readline()  # {"FPS": 29.96...}
        logger.debug(f"worker.init() message: {welcome_msg}")
        try:
            parsed = json.loads(welcome_msg)
            self.parsed_metadata = parsed
            self.FPS = parsed["FPS"]
            self.ok = True
        except:
            logger.warning("error occured while trying to parse file FPS")
            self.ok = False
        self.subscribed_messages: dict[str, Awaitable] = {}
        self.reader_task = asyncio.create_task(self._reciever_loop())
        return self.ok

    def register_channel_subscriber(
        self, channel_name: str, callback_coro: Awaitable[str]
    ):
        """registers a callback for messages starting with "channel_name"
        the communication from worker is of the form: channel_name.encode() + json.dumps(...).encode() + b"\n"
        callback will receive only the json.dumps(...) part

        Args:
            channel_name (str): name of message to subscribe
            callback_coro (Awaitable[str]): coroutine to be awaited when message arrives
        """
        self.subscribed_messages.update({channel_name: callback_coro})

    async def _reciever_loop(self):
        while True:
            try:
                msg = await self.reader.readline()
                if len(msg) == 0 and self.reader.at_eof():
                    logger.debug("reciever loop got EOF")
                    self.ok = False
                    await self.close()
                    # task should now be cancelled
                    return
                msg = msg.decode().rstrip()

                message_registered = False
                for m, coro in self.subscribed_messages.items():
                    if msg.startswith(m):
                        await coro(msg.lstrip(m))
                        logger.debug(
                            f'reciever loop got message "{m}", len(msg): {len(msg)}'
                        )
                        message_registered = True
                if not message_registered:
                    logger.debug(f"reciever loop got unexpected message: {msg}")
                # logger.debug(f"reciever loop got labels: {labels}")
            except asyncio.CancelledError:
                logger.debug("reciever loop got cancelled")
                return

    async def get_labels(self, start_frame: int, end_frame: int):
        """coroutine that request computation of labels for
        all frames in range from start_frame to end_frame.

        Args:
            start_frame (int): first frame index of which labels are requested
            end_frame (int): last frame index of which labels are requested
        """
        command = f"{start_frame}-{end_frame}" + "\n"
        logger.debug(f"get labels invoked with command: {command}")
        self.writer.write(command.encode())
        await self.writer.drain()

    async def start_writing_to_namedpipe(self, namedpipe: str, config: dict):
        """Request the worker to begin generation of the anonymized
        video given the config and pipe it to provided `named_pipe`

        Args:
            namedpipe (str): named pipe where worker should pipe the anonymized video
            config (dict): dict with optional fields:
                "treshold": detection treshold that the user set in browser,
                "shape": one of ["rectangle", "ellipse", "bbox"], type of anonymizing shape that the user set in browser,
                "background": on of ["blur", "pixelate", "black"], type of fill of anonymizing shape that the user set in browser,
                "preview-scores": bool switch that the user set in browser,

        """
        logger.debug(f"start_writing_to_namedpipe to pipe: {namedpipe}")
        msg = json.dumps({"namedpipe": namedpipe, "config": config})
        command = "serve under named_pipe: " + msg + "\n"
        self.writer.write(command.encode())
        await self.writer.drain()

    async def close(self):
        """gracefully close the worker connection"""
        self.reader_task.cancel()
        # await self.reader_task # TODO check if this line breaks anything if uncommented
        self.writer.close()  # writes EOF
        await self.writer.wait_closed()
        logger.debug(
            f"worker connection closed: file url: {self.url}, worker: {self.host}:{self.port}"
        )
