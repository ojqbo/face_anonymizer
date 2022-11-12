import argparse
import asyncio
import json
import logging
import subprocess
import typing
from pathlib import Path

from aiohttp import web

from face_anonymizer.src.clientFile import clientFile
from face_anonymizer.src.compute.clientComputeWorker import clientComputeHandler
from face_anonymizer.src.utils import unique_id_generator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
STATIC_ROOT_PATH = Path("./frontend/")


async def index(request: web.Request):
    return web.FileResponse(STATIC_ROOT_PATH / "index.html")


async def anonymized_fileobj_handle(request: web.Request) -> web.StreamResponse:
    """request handle to use with aiohttp server. Responds with video file
    that is anonymized dynamically. Does not accept range requests
    nor resumable downloads.

    Args:
        request (web.Request): request instance provided from aiohttp webserver.

    Returns:
        web.StreamResponse: response object expected by the aiohttp webserver.
    """
    response = web.StreamResponse()
    loop = asyncio.get_running_loop()
    name = request.match_info.get("name", "404")
    filename = request.match_info.get("filename", "404")
    response.headers.update(
        {"Content-Disposition": f"attachment; filename: {filename}"}
    )
    await response.prepare(request)
    if name not in request.app["namedpipeouts"]:
        return web.Response(text="404: Resource Not Found", status=404)
    fobj = request.app["namedpipeouts"][name]
    while True:
        try:
            b = await loop.run_in_executor(None, fobj.read, 64 * 1024)
        except ValueError as e:
            if e.args[0] == "read of closed file":
                logger.debug(
                    f"anonymized_fileobj_handle: namedpipe closed, signal: {e}"
                )
                break
            else:
                raise e
        if len(b) == 0:
            break
        try:
            await response.write(b)
        except ConnectionResetError as e:
            fobj.close()
            logger.debug(f"anonymized_fileobj_handle: download stopped, signal: {e}")
            break  # user aborted download
    return response


async def wshandle(request: web.Request) -> web.WebSocketResponse:
    """request handle to use with aiohttp server. Establishes WebSocket
    communication channel, and handles the messages.

    WebSocket exchanged message types:
    all text-type messages are expected to be json serialized dicts
    with one of the dict fields is required to be "msg":
        "msg" types from client to server:
            "file available", user has provided a file to anonymize:
                other fields will contain keys:
                    "name" with filename of the video,
                    "size" with total size of the video file (int),
                    "type" with mime-type of the file,
            "get", a request for labels for a range of video frames:
                other fields will contain keys:
                    "from" start-index of the range of frames
                        for which labels to calculate
                    "upto" end-index of the range of frames
                        for which labels to calculate
            "user config, request download", request to generate the anonymized file:
                other fields are optional with keys:
                    "threshold": detection threshold that the user set in browser,
                    "shape": one of ["rectangle", "ellipse", "bbox"],
                        type of anonymizing shape that the user set in browser,
                    "background": on of ["blur", "pixelate", "black"],
                        type of fill of anonymizing shape that the user set in browser,
                    "preview-scores": bool switch that the user set in browser,
        "msg" types from server to client:
            "new file response", response to file upload by the client:
                other fields will contain keys:
                    "FPS" with float value
            "lab", response with labels mapped to indexes of video frames:
                other fields will contain keys:
                    "lab" with a dict[int: list[list[float]]] of {frame_index:labels}
            "get", request for the client to provide slice of the uploaded
                file, i.e. file_bytes[S:E]:
                other fields will contain keys:
                    "S" with int value equal to start-index of requested slice (incl.)
                    "E" with int value equal to end-index of requested slice (excl.)
            "download ready", notification that generated on the fly anonymized file
                is available to download:
                other fields will contain keys:
                    "path" with endpoint on which the file may be downloaded
            "progress", notification of the approximate progress of download:
                other fields will contain keys:
                    "estimated_time_left" estimated remaining download time in seconds
                    "ratio_done" value in (0,1) range of approximate download progress
    there is only a single binary-type message from client to server containing
    user file slice of bytes from "S" to "E", preceded by 8-byte integer,
    big-endian ordered (byteorder='big'), with value equal to the index
    of starting byte "S". No binary message is send to the client over WebSocket.

    Args:
        request (web.Request): request instance provided from aiohttp webserver.

    Returns:
        web.WebSocketResponse: response object expected by the aiohttp webserver.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"new websocket client connected: request.remote: {request.remote}")
    tasks: list[asyncio.Task] = []

    resource_name = None

    async def wshandle_own_cleanup():
        [t.cancel() for t in tasks]
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        popped_client_file: Path = request.app["filepaths"].pop(resource_name, [])
        anonymized_p_fobj: typing.IO[bytes] = request.app["namedpipeouts"].pop(
            resource_name, None
        )
        popped_w = request.app["workers"].pop(resource_name, None)
        for p in popped_client_file:
            p.unlink()
        if popped_w is not None:
            await popped_w.close()
        logger.debug(
            f"wshandle_own_cleanup invoked, resource_name: {resource_name} "
            f"popped: client_file:{popped_client_file != None} "
            f"fobj_anon:{anonymized_p_fobj != None} "
            f"worker:{popped_w != None} "
            f"len(tasks): {len(tasks)}"
        )

    async for msg in ws:
        await asyncio.gather(*[t for t in tasks if t.done()])
        tasks = [
            t for t in tasks if not t.done()
        ]  # remove references to finished tasks
        if msg.type == web.WSMsgType.text:
            logger.debug(f"text arrived: {msg}")
            # sanitize
            try:
                parsed_msg = json.loads(msg.data)
                msg_about = parsed_msg["msg"]
            except json.JSONDecodeError:
                logger.debug(
                    "parsing failed or the message had no msg property,"
                    f" ignoring (from {request.remote})"
                )
                continue
            logger.debug(f"client {request.remote}: parsed_msg: {parsed_msg}")

            if msg_about == "file available":
                logger.info(f"client {request.remote} starts serving new file")
                await wshandle_own_cleanup()
                # if resource_name is not None:
                #     del fileobjects[resource_name]
                #     del fileobjects[resource_name]
                resource_name = unique_id_generator()
                filehandler = clientFile(ws, parsed_msg)

                async def save_file(path: str | Path):
                    logger.debug(f"saving client file to {path}")
                    loop = asyncio.get_running_loop()
                    with open(path, "wb") as f:
                        request.app["filepaths"][resource_name] += [client_file_path]
                        filehandler.seek(0)
                        while True:
                            b = await filehandler.read(1024 * 1024)
                            if len(b) == 0:
                                break
                            await loop.run_in_executor(None, f.write, b)

                request.app["filepaths"][resource_name] = []
                client_file_path = Path(f"./client_files/{resource_name}")
                client_file_path.parent.mkdir(exist_ok=True, parents=True)

                async def task():
                    await save_file(client_file_path)
                    logger.debug(f"client file saved under {client_file_path}")
                    worker = clientComputeHandler(ws, client_file_path)
                    logger.debug("worker instance ready")

                    await worker.start()
                    logger.debug("worker started")
                    if worker.ok is True:
                        request.app["workers"][resource_name] = worker
                    else:
                        pass

                tasks.append(asyncio.create_task(task()))
            elif msg_about == "get":  # get labels

                async def task():
                    await request.app["workers"][resource_name].request_label_range(
                        parsed_msg["from"], parsed_msg["upto"]
                    )

                tasks.append(asyncio.create_task(task()))
            elif msg_about == "user config, request download":
                # async def task():
                valid_keys = [
                    "threshold",
                    "shape",
                    "background",
                    "preview-scores",
                ]
                config = {k: parsed_msg[k] for k in valid_keys if k in parsed_msg}
                # prepare a pipeline that generates anonymized video:

                async def task():
                    extra_uid = unique_id_generator(11)
                    pipename_video = Path(
                        f"pipes/{resource_name}_{extra_uid}"
                    ).absolute()
                    pipename_video.parent.mkdir(exist_ok=True, parents=True)
                    subprocess.run(
                        f"mkfifo {pipename_video}",
                        shell=True,
                    )

                    await request.app["workers"][resource_name].serve_under_named_pipe(
                        pipename_video, config
                    )
                    request.app["namedpipeouts"][resource_name] = subprocess.Popen(
                        ["cat", f"{pipename_video}"], stdout=subprocess.PIPE
                    ).stdout
                    request.app["filepaths"][resource_name] += [pipename_video]
                    logger.debug("preparing anonymized video export: pipeline ready")
                    await ws.send_str(
                        json.dumps(
                            {
                                "msg": "download ready",
                                "path": f"/anonymized/{resource_name}/{extra_uid}",
                            }
                        )
                    )
                    logger.debug(
                        "anonymized file should be available under "
                        f"/anonymized/{resource_name}/{extra_uid}"
                    )

                tasks.append(asyncio.create_task(task()))
            else:
                logger.debug(
                    f"msg property not recognized, ignoring (from {request.remote})"
                )
        elif msg.type == web.WSMsgType.binary:
            logger.info(f"binary data arrived, len: {len(msg.data)} bytes")
            if filehandler is None:
                logger.info("binary data arrived, but filehandler is None")
            else:
                filehandler.receive_data(msg.data)
        elif msg.type == web.WSMsgType.close:
            logger.debug("new msg: msg.type == web.WSMsgType.close")
            break
    logger.info(f"websocket end loop, request.remote: {request.remote}")
    await wshandle_own_cleanup()
    return ws


async def shutdown_callback(app: web.Application):
    """coroutine that cleans up any opened files or connections on application close

    Args:
        app (web.Application): aiohttp server application
    """
    # [opt] placeholder for cleanup
    for uid, filepaths in app["filepaths"].items():
        logger.debug(f"unlinking {filepaths}")
        for filepath in filepaths:
            filepath.unlink()  # delete user file/namedpipe
    logger.info("shutdown_callback: graceful shutdown OK")


routes = [
    web.get(
        "/anonymized/{name}/{nth_attempt_uid}/{filename}", anonymized_fileobj_handle
    ),
    web.get("/ws", wshandle),
    web.get("/", index),
    web.static("/", STATIC_ROOT_PATH, append_version=True),
]


def make_app():
    app = web.Application()
    app.add_routes(routes)
    app.on_shutdown.append(shutdown_callback)
    app["filepaths"]: dict[str, Path] = {
        # "file_hash": path_to_saved_file
    }
    app["namedpipeouts"]: dict[str, typing.IO[bytes]] = {
        # "file_hash": subprocess.stdout.buffer
    }
    app["workers"]: dict[str, clientComputeHandler] = {
        # "file_hash": workerConnection instance
    }
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Web app for video anonymization.",
        epilog="The app is based on aiohttp.",
    )
    parser.add_argument(
        "--port",
        metavar="port",
        type=int,
        default=8080,
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

    web.run_app(make_app(), host=args.interface, port=args.port)
