from aiohttp import web
from aiohttp.typedefs import LooseHeaders
from aiohttp.abc import AbstractStreamWriter
from aiohttp import hdrs
from typing import (
    IO,
    Any,
    Optional,
    cast,
)
from aiohttp.web_exceptions import (
    HTTPNotModified,
    HTTPPartialContent,
    HTTPPreconditionFailed,
    HTTPRequestRangeNotSatisfiable,
)
import mimetypes
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


class FileObjectResponse(web.FileResponse):
    """A response object to send files based on [python] file like objects with .size property."""

    def __init__(
        self,
        fobj: IO[Any],
        chunk_size: int = 1 * 1024 * 1024 - 8,  # 256 * 1024,
        status: int = 200,
        reason: Optional[str] = None,
        headers: Optional[LooseHeaders] = None,
    ) -> None:
        """Creates a response object compatible with aiohttp
        to send files based on [python] file like objects with .size property,
        that supports .read() and .seek() calls. Can be used with range-requests
        (partial or reusumable downloads). Intended usage is to return this
        object in request handle when using aiohttp server. Is used to send large files.

        Example usage with aiohttp:
            >>> from aiohttp import web
            >>> fobj = open("a_1kb_file.txt")
            >>> fobj.size = 1024
            >>> async def fileobj_handle(request: web.BaseRequest):
            >>>     return FileObjectResponse(foobj)
            >>> app = web.Application()
            >>> app.add_routes([web.get("/file.txt", fileobj_handle)])
            >>> app.on_shutdown.append(fobj.close)
            >>> web.run_app(app)


        Args:
            fobj (IO[Any]): file like object with .size property
            chunk_size (int, optional): limit to the size of a single chunk
                when responding to client. Large files are send in chunks of this size.
                Defaults to 1*1024*1024-8.
            status (int, optional): HTTP Status, like 404 if file not found. Defaults to 200.
            reason (Optional[str], optional): HTTP Reason, like "Not Found" for 404 status.
                When None, gets autocalculated based on status. Just leave it as None.
                Put some numbers in the beginning to cause debug mayhem. Defaults to None.
            headers (Optional[LooseHeaders], optional): CIMultiDict (dict with case insensitive keys)
                instance for outgoing HTTP headers. Defaults to None.
        """
        super().__init__(
            "_no_path",
            chunk_size=chunk_size,
            status=status,
            reason=reason,
            headers=headers,
        )

        self._fobj = fobj  # must be in open state

    async def prepare(self, request: web.BaseRequest) -> Optional[AbstractStreamWriter]:
        """Analyzes the request and sends the file.

        Args:
            request (web.BaseRequest): aiohttp web server response

        Returns:
            Optional[AbstractStreamWriter]: writer that is done sending file-chunks.
        """
        fake_filepath = (
            self._fobj.fake_filename
            if hasattr(self._fobj, "fake_filename")
            else self._fobj.__name__
        )

        loop = asyncio.get_event_loop()

        status = self._status
        if hasattr(self._fobj, "size"):
            file_size = self._fobj.size
        else:
            print("== BEGIN NOTE: ==")
            print(
                "class FileObjectResponse is not ment to be used with vanilla filehandlers"
            )
            print(
                "fobj should have .size property. For now, whole file size is set to len(fobj.read())"
            )
            print("running:")
            print("  file_size = len(self._fobj.read())")
            print("  self._fobj.seek(0)")
            print("instead of:")
            print("  file_size = self._fobj.size")
            print("Reading whole file by .read() is not what you'd want")
            print("== END NOTE ==")
            file_size = len(self._fobj.read())
            self._fobj.seek(0)
        count = file_size

        start = None

        ifrange = request.if_range
        if ifrange is None:
            ## or if (st.st_mtime <= ifrange.timestamp()): ## st (file stats) not available
            # If-Range header check:
            # condition = cached date >= last modification date
            # return 206 if True else 200.
            # if False:
            #   Range header would not be processed, return 200
            # if True but Range header missing
            #   return 200
            try:
                rng = request.http_range
                start = rng.start
                end = rng.stop
            except ValueError:
                # https://tools.ietf.org/html/rfc7233:
                # A server generating a 416 (Range Not Satisfiable) response to
                # a byte-range request SHOULD send a Content-Range header field
                # with an unsatisfied-range value.
                # The complete-length in a 416 response indicates the current
                # length of the selected representation.
                #
                # Will do the same below. Many servers ignore this and do not
                # send a Content-Range header with HTTP 416
                self.headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)

            # If a range request has been made, convert start, end slice
            # notation into file pointer offset and count
            if start is not None or end is not None:
                if start < 0 and end is None:  # return tail of file
                    start += file_size
                    if start < 0:
                        # if Range:bytes=-1000 in request header but file size
                        # is only 200, there would be trouble without this
                        start = 0
                    count = file_size - start
                else:
                    # rfc7233:If the last-byte-pos value is
                    # absent, or if the value is greater than or equal to
                    # the current length of the representation data,
                    # the byte range is interpreted as the remainder
                    # of the representation (i.e., the server replaces the
                    # value of last-byte-pos with a value that is one less than
                    # the current length of the selected representation).
                    count = (
                        min(end if end is not None else file_size, file_size) - start
                    )

                if start >= file_size:
                    # HTTP 416 should be returned in this case.
                    #
                    # According to https://tools.ietf.org/html/rfc7233:
                    # If a valid byte-range-set includes at least one
                    # byte-range-spec with a first-byte-pos that is less than
                    # the current length of the representation, or at least one
                    # suffix-byte-range-spec with a non-zero suffix-length,
                    # then the byte-range-set is satisfiable. Otherwise, the
                    # byte-range-set is unsatisfiable.
                    self.headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                    self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                    return await super().prepare(request)

                status = HTTPPartialContent.status_code
                # Even though you are sending the whole file, you should still
                # return a HTTP 206 for a Range request.
                self.set_status(status)

        self.content_type = "application/octet-stream"
        self.content_length = count
        self.headers[hdrs.ACCEPT_RANGES] = "bytes"

        real_start = cast(int, start)

        if status == HTTPPartialContent.status_code:
            self.headers[hdrs.CONTENT_RANGE] = "bytes {}-{}/{}".format(
                real_start, real_start + count - 1, file_size
            )

        if request.method == hdrs.METH_HEAD or self.status in [204, 304]:
            return await super().prepare(request)

        fobj = self._fobj  ## await loop.run_in_executor(None, filepath.open, "rb")
        if start:  # be aware that start could be None or int=0 here.
            offset = start
        else:
            offset = 0
        try:
            return await self._sendfile(request, fobj, offset, count)
        finally:
            pass  ## this class does not close the fobj, close it on websocket closure or elsewhere
            ## await loop.run_in_executor(None, fobj.close)

    async def _sendfile(
        self, request: web.BaseRequest, fobj: IO[Any], offset: int, count: int
    ) -> AbstractStreamWriter:
        """seeks the file to offset, chunks requested range, and sends (chunked)
        to writer which is created based on request

        Args:
            request (web.BaseRequest): client request
            fobj (IO[Any]): file like object in opened state
            offset (int): offset to seek the file
            count (int): number of bytes to read from offset

        Returns:
            AbstractStreamWriter: writer that is done sending file-chunks.
        """
        writer = await web.StreamResponse.prepare(self, request)
        assert writer is not None

        logger.debug(
            f"_sendfile(): offset: {offset}, count: {count}, self._chunk_size: {self._chunk_size}"
        )
        # To keep memory usage low,fobj is transferred in chunks
        # controlled by the constructor's chunk_size argument.

        chunk_size = self._chunk_size
        # loop = asyncio.get_event_loop()

        # await loop.run_in_executor(None, fobj.seek, offset)
        fobj.seek(offset)

        # chunk = await loop.run_in_executor(None, fobj.read, chunk_size)
        chunk = await fobj.read(chunk_size)
        while chunk:
            await writer.write(chunk)
            count = count - chunk_size
            if count <= 0:
                break
            # chunk = await loop.run_in_executor(None, fobj.read, min(chunk_size, count))
            chunk = await fobj.read(min(chunk_size, count))

        await writer.drain()
        return writer
