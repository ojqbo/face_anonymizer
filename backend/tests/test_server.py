import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import pytest_aiohttp.plugin  # type: ignore
from aiohttp import web

from backend.server import make_app


@pytest.fixture
def index_content() -> str:
    return Path("frontend/index.html").read_text()


@pytest.mark.asyncio
async def test_index(
    aiohttp_client: pytest_aiohttp.plugin.aiohttp_client, index_content: str
):
    cli = await aiohttp_client(make_app())
    resp = await cli.get("/")
    assert resp.status == 200
    assert await resp.text() == index_content


@pytest.mark.asyncio
async def test_websocketAPI(
    aiohttp_client: pytest_aiohttp.plugin.aiohttp_client,
    video_file: str,
    video_raw_frames: np.ndarray,
):
    cli = await aiohttp_client(make_app())
    ws_conn = await cli.ws_connect("/ws")
    video_file_bytes = Path(video_file).read_bytes()

    # pretend that user selected a file in browser
    await ws_conn.send_json(
        {
            "msg": "file available",
            "name": video_file,
            "size": len(video_file_bytes),
            "type": "video/mp4",
        }
    )
    # now upload should start, i.e. server starts requesting file slices
    while True:
        msg = await ws_conn.receive()
        assert msg.type == web.WSMsgType.text
        parsed_msg = msg.json()
        assert "msg" in parsed_msg
        if parsed_msg["msg"] == "new file response":
            break
        assert parsed_msg["msg"] == "get"
        resp_slice = video_file_bytes[parsed_msg["S"] : parsed_msg["E"]]
        start_int64 = parsed_msg["S"].to_bytes(length=8, byteorder="big")
        await ws_conn.send_bytes(start_int64 + resp_slice)
    # upload done, server should response that further processing is ready
    # and tell the FPS rate of the uploaded video
    assert parsed_msg["msg"] == "new file response"
    fps_from_filename = float(str(video_file).split("fps=")[1].split(".")[0])
    assert "FPS" in parsed_msg
    try:
        float(parsed_msg["FPS"])
    except ValueError:
        assert False, "parsed FPS field could not be converted to float"
    assert parsed_msg["FPS"] == fps_from_filename

    # test request for labels
    # get only the first label
    await ws_conn.send_json(
        {
            "msg": "get",
            "from": 0,
            "upto": 0,
        }
    )
    msg = await ws_conn.receive()
    parsed_msg = msg.json()
    assert "msg" in parsed_msg
    assert parsed_msg["msg"] == "lab"
    assert {"0"} == set(parsed_msg["lab"])
    assert isinstance(parsed_msg["lab"]["0"], list)

    # request labels with video seek-ing
    received_labels = {}
    requested_labels = set()
    for F, T in [(0, 2), (10, 16), (4, 6)]:
        await ws_conn.send_json(
            {
                "msg": "get",
                "from": F,
                "upto": T,
            }
        )
        requested_labels |= set([str(i) for i in range(F, T + 1)])
        timeout_timestamp = 4 + time.time()
        while time.time() <= timeout_timestamp:
            msg = await ws_conn.receive()
            parsed_msg = msg.json()
            received_labels.update(parsed_msg["lab"])
            assert "msg" in parsed_msg
            assert parsed_msg["msg"] == "lab"
            assert set(parsed_msg["lab"]) <= requested_labels
            if set(received_labels) == requested_labels:
                break
        else:
            assert False, "receive labels timeout, or labels were wrong"
    # assertion below is redundant. while exits only if this condition is true
    assert set(received_labels) == requested_labels

    # test exporting video
    await ws_conn.send_json(
        {
            "msg": "user config, request download",
            "threshold": 0.3,
            "shape": "bbox",
            "preview-scores": True,
        }
    )
    msg = await ws_conn.receive()
    parsed_msg = msg.json()
    assert "msg" in parsed_msg
    assert parsed_msg["msg"] == "download ready"
    download_uri = parsed_msg["path"]
    anon_filepath = f"{video_file}_anon.mp4"
    dn_resp = await cli.get(download_uri)
    assert dn_resp.status == 200
    anonymized_video_bytes = await dn_resp.read()
    Path(anon_filepath).write_bytes(anonymized_video_bytes)

    # check if the exported video has the right format
    frames_exported = int(
        subprocess.check_output(
            f"ffprobe -v error -select_streams v:0"
            f" -count_packets -show_entries stream=nb_read_packets"
            f" -of csv=p=0 {anon_filepath}",
            shell=True,
        )
    )
    w, h, fps = (
        subprocess.check_output(
            f"ffprobe -v error -select_streams v:0"
            f" -show_entries stream=width,height,avg_frame_rate"
            f" -of csv=p=0 {anon_filepath}",
            shell=True,
        )
        .decode()
        .split(",")
    )
    assert frames_exported == len(video_raw_frames)
    assert fps_from_filename == int(fps.split("/")[0]) / int(fps.split("/")[1])
    assert int(w) == video_raw_frames.shape[-2]
    assert int(h) == video_raw_frames.shape[-3]
