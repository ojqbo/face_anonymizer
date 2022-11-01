import asyncio
import json
import subprocess

import numpy as np
import pytest

from backend.src.compute.clientComputeWorker import clientComputeHandler

from .dummyWebsocketClient import dummyWebsocketClient


@pytest.mark.asyncio
async def test_clientComputeWorker_videoExport(
    video_raw_frames: np.ndarray, video_file: str, dummy_model
):
    num_of_tasks_bfr_anything = len(asyncio.all_tasks())
    fps_from_filename = float(str(video_file).split("fps=")[1].split(".")[0])
    ws = dummyWebsocketClient(video_file)
    worker = clientComputeHandler(ws, video_file, dummy_model)  # type: ignore
    # ws is expected to be WebSocket

    # check if start creates background task
    num_of_tasks = len(asyncio.all_tasks())
    await worker.start()
    assert (num_of_tasks + 3) == len(asyncio.all_tasks())

    # check if start returns proper weclome message
    parsed_msg = json.loads(ws.received_messages[-1])
    assert "msg" in parsed_msg
    assert parsed_msg["msg"] == "new file response"
    assert "FPS" in parsed_msg
    try:
        float(parsed_msg["FPS"])
    except ValueError:
        assert False, "parsed FPS field could not be converted to float"

    # check video exporting:
    anon_filepath = f"{video_file}_anon.mp4"
    num_of_tasks = len(asyncio.all_tasks())
    await worker.serve_under_named_pipe(
        namedpipe=anon_filepath,
        config={
            "threshold": 0.3,
            "shape": "bbox",
            "preview-scores": True,
        },
    )
    # it is unknown how long it will take to generate the output file
    if worker._serve_file_task is not None:
        await worker._serve_file_task  # note: normally this task isn't awaited
        # it is allowed for this task to be None if done serving
        # however at the moment _serve_file_task is not being reset to None.
        # Only when next serve_under_named_pipe invoked, the task gets replaced
    assert num_of_tasks == len(asyncio.all_tasks())

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

    num_of_tasks = len(asyncio.all_tasks())
    await worker.close()
    assert (num_of_tasks - 3) == len(asyncio.all_tasks())
    assert num_of_tasks_bfr_anything == len(asyncio.all_tasks())


@pytest.mark.asyncio
async def test_clientComputeWorker_interactive(
    video_raw_frames: np.ndarray, video_file: str, dummy_model
):
    num_of_tasks_bfr_anything = len(asyncio.all_tasks())
    ws = dummyWebsocketClient(video_file)
    worker = clientComputeHandler(ws, video_file, dummy_model)  # type: ignore
    # ws is expected to be WebSocket
    await worker.start()

    def extract_labels_from_msgs_list(msg_list):
        parsed_msgs = [json.loads(m) for m in msg_list]
        parsed_msgs = [m for m in parsed_msgs if m["msg"] == "lab"]
        all_recieved_labels = {}
        for m in parsed_msgs:
            all_recieved_labels.update({int(k): v for k, v in m["lab"].items()})
        return all_recieved_labels

    # check label-range requests
    await worker.request_label_range(0, 3)
    # it is unknown how long it will take to generate the labels
    while worker._labels_to_send_queue.qsize():
        await asyncio.sleep(0.1)
    assert {0, 1, 2, 3} == set(extract_labels_from_msgs_list(ws.received_messages))
    ws.received_messages = []  # reset msg history
    # simulate another request for label range
    await worker.request_label_range(1, 2)
    while worker._labels_to_send_queue.qsize():
        await asyncio.sleep(0.1)
    assert {1, 2} == set(extract_labels_from_msgs_list(ws.received_messages))
    ws.received_messages = []  # reset msg history
    # simulate another request for label range
    await worker.request_label_range(5, 9)
    while worker._labels_to_send_queue.qsize():
        await asyncio.sleep(0.1)
    assert {5, 6, 7, 8, 9} == set(extract_labels_from_msgs_list(ws.received_messages))

    await worker.close()
    assert num_of_tasks_bfr_anything == len(asyncio.all_tasks())
