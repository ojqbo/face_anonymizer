#!/usr/bin/python3
from aiohttp import web
import asyncio
import argparse
import json
from pathlib import Path
import typing
from src.FileObjectResponse import FileObjectResponse
from src.ExposeClientFile import MountedClientFile
from src.clientFile import clientFile
from src.utils import unique_id_generator
from src.OpenCVmiddleware import workerConnection
import configs
import subprocess
import logging

# TODO cleanup unused pipes

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
STATIC_ROOT_PATH = Path("../frontend/")


async def index(request: web.BaseRequest):
    return web.FileResponse(STATIC_ROOT_PATH / "index.html")


async def dynamic_fileobj_handle(request: web.BaseRequest) -> web.Response:
    """request handle to use with aiohttp server. Responds with file
    mounted by the user in his browser. Accepts range requests.

    Args:
        request (web.BaseRequest): request instance provided from aiohttp webserver.

    Returns:
        web.Response: response object expected by the aiohttp webserver.
    """
    name = request.match_info.get("name", "404")
    if name in request.app["fileobjects"]:
        # name is designed to be at least 128 bits long
        # it's cryptographically safe to assume that any adversarial user will not find any other user's video
        # !note that at the moment the rng used is not cryptographically safe!
        return FileObjectResponse(request.app["fileobjects"][name])
    return web.Response(text="404: Resource Not Found", status=404)


async def anonymized_fileobj_handle(request: web.BaseRequest) -> web.StreamResponse:
    """request handle to use with aiohttp server. Responds with video file
    that is anonymized dynamically. Does not accept range requests
    nor reusumable downloads.

    Args:
        request (web.BaseRequest): request instance provided from aiohttp webserver.

    Returns:
        web.StreamResponse: response object expected by the aiohttp webserver.
    """
    response = web.StreamResponse()
    loop = asyncio.get_running_loop()
    name = request.match_info.get("name", "404")
    await response.prepare(request)
    if name not in request.app["fileobjects"]:
        return web.Response(text="404: Resource Not Found", status=404)
    fobj = request.app["fileobjects"][name]
    while True:
        b = await loop.run_in_executor(None, fobj.read, 64 * 1024)
        if len(b) == 0:
            break
        await response.write(b)
    return response


async def wshandle(request: web.BaseRequest) -> web.WebSocketResponse:
    """request handle to use with aiohttp server. Establishes WebSocket
    communication channel, and handles the messages.

    WebSocket exchanged message types:
    all text-type messages are expected to be json serialized dicts
    with one of the dict fields is required to be "msg":
        "msg" types from client to server:
            "file available", information that the user has provided a file to anonymize:
                other fields will contain keys:
                    "name" with filename of the video,
                    "size" with total size of the video file (int),
                    "type" with mime-type of the file,
            "get", a request for labels for a range of video frames:
                other fields will contain keys:
                    "from" with start-index of the range of frames for which labels to calculate
                    "upto" with end-index of the range of frames for which labels to calculate
            "user config, request download", request to generate the anonymized file:
                other fields will contain keys:
                    "treshold": detection treshold that the user set in browser,
                    "shape": one of ["rectangle", "ellipse", "bbox"], type of anonymizing shape that the user set in browser,
                    "background": on of ["blur", "pixelate", "black"], type of fill of anonymizing shape that the user set in browser,
                    "preview-scores": bool switch that the user set in browser,
        "msg" types from server to client:
            "new file response", response to file upload by the client:
                other fields will contain keys:
                    "config" with suggested hint-fields for tweaking anonymization
                    "FPS" with float value
            "lab", response with labels mapped to indexes of video frames:
                other fields will contain keys:
                    "lab" with a dict[int: list[list[float]]] with mapping of frame_index:frame_labels
            "get", request for the client to provide slice of the uploaded file file_bytes[S:E]:
                other fields will contain keys:
                    "S" with int value equal to start-index of requested slice
                    "E" with int value equal to end-index of requested slice
            "download ready", notification that generated on the fly anonymized file is available to download:
                other fields will contain keys:
                    "path" with endpoint on which the file may be downloaded
            "progress", notification of the approximate progress of download:
                other fields will contain keys:
                    "estimated_time_left" with estimated remaining download time in seconds
                    "ratio_done" with value in (0,1) range of approximate download progress
    there is only a single binary-type message from client to server containing
    user file slice of bytes from "S" to "E", preceded by 8-byte integer,
    big-endian ordered (byteorder='big'), with value equal to the index
    of starting byte "S". No binary message is send to the client over WebSocket.

    Args:
        request (web.BaseRequest): request instance provided from aiohttp webserver.

    Returns:
        web.WebSocketResponse: response object expected by the aiohttp webserver.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"new websocket client connected: request.remote: {request.remote}")
    tasks = []

    resource_name = None
    anonymized_resource_name = None

    async def wshandle_own_cleanup():
        popped_fobj = request.app["fileobjects"].pop(resource_name, None)
        anonymized_p_fobj = request.app["fileobjects"].pop(
            anonymized_resource_name, None
        )
        popped_w = request.app["workers"].pop(resource_name, None)
        await asyncio.gather(*tasks)
        if popped_w is not None:
            await popped_w.close()
        logger.debug(
            f"wshandle_own_cleanup invoked, resource_name: {resource_name} "
            f"poped: fobj:{popped_fobj != None} fobj_anon:{anonymized_p_fobj != None} worker:{popped_w != None} "
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
                # TODO: validate message
                msg_about = parsed_msg["msg"]
            except:
                logger.debug(
                    "parsing failed or the message had no msg property, ignoring (from {request.remote})"
                )
                continue
            logger.debug(f"client {request.remote}: parsed_msg: {parsed_msg}")

            if msg_about == "file available":
                logger.info(f"client {request.remote} starts serving new file")
                await wshandle_own_cleanup()
                # if resource_name is not None:
                #     del fileobjects[resource_name]
                #     del fileobjects[resource_name]
                filehandler = clientFile(ws, parsed_msg)
                resource_name = unique_id_generator()
                test = MountedClientFile(filehandler, resource_name)
                await test.start()
                request.app["fileobjects"][resource_name] = filehandler
                handle_path = f"/dynamic/{resource_name}"
                request.app["workers"][resource_name] = workerConnection(
                    f"http://APP_HOST:{APP_PORT}{handle_path}", WORKER_HOST, WORKER_PORT
                )

                async def task():
                    async def got_labels(l: str):
                        await ws.send_str('{"msg":"lab","lab":' + l + "}")

                    await app["workers"][resource_name].init()
                    request.app["workers"][resource_name].register_channel_subscriber(
                        channel_name="labels", callback_coro=got_labels
                    )
                    if request.app["workers"][resource_name].ok == False:
                        # TODO what to do in this case?
                        pass
                    await ws.send_str(
                        json.dumps(
                            {
                                "msg": "new file response",
                                "path": handle_path,
                                "config": configs.defaultClientConfig,  # config will also land in window.default_pipeline_config
                                **request.app["workers"][
                                    resource_name
                                ].parsed_metadata,  # "FPS", [opt]"total frames"
                            }
                        )
                    )

                tasks.append(asyncio.create_task(task()))
            elif msg_about == "get":  # get labels

                async def task():
                    # worker will respond using registered "labels" message subscriber when labels are ready
                    await request.app["workers"][resource_name].get_labels(
                        parsed_msg["from"], parsed_msg["upto"]
                    )

                tasks.append(asyncio.create_task(task()))
            elif msg_about == "user config, request download":
                # async def task():
                valid_keys = [
                    "treshold",
                    "shape",
                    "background",
                    "preview-scores",
                ]
                config = {k: parsed_msg[k] for k in valid_keys}
                # logging.error(f"download requests not yet implemented, config: {config}")
                # prepare a pipeline
                anonymized_resource_name = f"{resource_name}_anonymized"

                async def task():
                    pipename_video = Path(
                        "pipes/" + anonymized_resource_name
                    ).absolute()
                    pipename_video.parent.mkdir(exist_ok=True, parents=True)
                    pipename_video = str(pipename_video)
                    if Path(pipename_video).exists():
                        subprocess.run(
                            f"rm {pipename_video}",
                            shell=True,
                        )
                    subprocess.run(
                        f"mkfifo {pipename_video}",
                        shell=True,
                    )
                    logger.debug(
                        f"1/3 of preparing a pipeline: mkfifo {pipename_video} done"
                    )

                    async def notify_progress_bar(r: str):
                        resp = {
                            "msg": "progress",
                            **json.loads(r),
                        }
                        await ws.send_str(json.dumps(resp))

                    request.app["workers"][resource_name].register_channel_subscriber(
                        channel_name="progress", callback_coro=notify_progress_bar
                    )
                    await request.app["workers"][resource_name].start_writing_to_namedpipe(
                        pipename_video, config
                    )
                    logger.debug(
                        f"2/3 of preparing a pipeline: writing_to_namedpipe initialized"
                    )
                    request.app["fileobjects"][anonymized_resource_name] = subprocess.Popen(
                        ["cat", f"{pipename_video}"], stdout=subprocess.PIPE
                    ).stdout
                    # fileobjects[anonymized_resource_name] = open(pipename_video) # blocking op - hangs the code. how to avoid $cat named_pipe > subprocess.stdout hack?
                    logger.debug(f"3/3 of preparing a pipeline: pipeline ready")
                    await ws.send_str(
                        json.dumps(
                            {
                                "msg": "download ready",
                                "path": f"/anonymized/{anonymized_resource_name}",
                                # "path": f"/dynamic/{resource_name}",
                            }
                        )
                    )
                    logger.debug(
                        f"anonymized file should be available under /anonymized/{anonymized_resource_name}"
                    )

                tasks.append(asyncio.create_task(task()))
            else:
                logger.debug(
                    f"msg proporty not recognized, ignoring (from {request.remote})"
                )
        elif msg.type == web.WSMsgType.binary:
            logger.info(f"binary data arrived, len: {len(msg.data)} bytes")
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
    logger.info(f"shutdown_callback: gracefull shutdown OK")


routes = [
    web.get("/dynamic/{name}", dynamic_fileobj_handle),  # for in-app use
    web.get("/anonymized/{name}", anonymized_fileobj_handle),  # for user
    # web.get('/test', test),
    web.get("/ws", wshandle),
    web.get("/", index),
    web.static("/", STATIC_ROOT_PATH, append_version=True),
]

app = web.Application()
app.add_routes(routes)
app.on_shutdown.append(shutdown_callback)
app["fileobjects"]: dict[str, typing.IO] = {
    # "file_hash": file-like-object
}
app["workers"]: dict[str, workerConnection] = {
    # "file_hash": workerConnection instance
}


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

    parser.add_argument(
        "--worker_port",
        metavar="13263",
        type=int,
        default=13263,
        help="port on which the server is started",
    )
    parser.add_argument(
        "--worker_host",
        metavar="localhost",
        type=str,
        default="localhost",
        help="port on which the server is started",
    )

    args = parser.parse_args()
    # update global variables
    APP_HOST, APP_PORT = args.interface, args.port
    WORKER_HOST, WORKER_PORT = args.worker_host, args.worker_port

    web.run_app(app, host=APP_HOST, port=APP_PORT)
