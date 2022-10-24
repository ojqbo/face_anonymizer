#!/usr/bin/python3
import asyncio
import argparse
from clientComputeWorker import clientComputeHandler
import gc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    H = clientComputeHandler(reader, writer)
    await H.start()  # ends at connection close (EOF from reader)
    await H.close()
    writer.close()
    del H
    gc.collect()


async def main(port: int):
    server = await asyncio.start_server(handle_client, "0.0.0.0", port)

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info(f"Serving on {addrs}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video processing server.",
        epilog="the following commands are accepted: int, int-int, get cache, save cache.",
    )
    parser.add_argument(
        "--port",
        metavar="port",
        type=int,
        default=13263,
        help="port on which the server is started",
    )
    parser.add_argument(
        "--interface",
        metavar="loopback",
        type=str,
        default="0.0.0.0",
        help="port on which the server is started",
    )

    args = parser.parse_args()
    asyncio.run(main(port=args.port))
