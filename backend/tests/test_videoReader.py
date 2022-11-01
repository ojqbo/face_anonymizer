import asyncio

import numpy as np
import pytest

from backend.src.compute.videoReader import videoReader


@pytest.mark.asyncio
async def test_videoReader(video_raw_frames: np.ndarray, video_file: str):
    reader_queue_size = 10
    reader = videoReader(video_src=video_file, frames_queue_size=reader_queue_size)

    # check if start creates background task
    num_of_tasks = len(asyncio.all_tasks())
    await reader.start()
    assert (num_of_tasks + 1) == len(asyncio.all_tasks())

    # check pop_frame() interface
    extracted_frames = []
    while True:
        f = await reader.pop_frame()
        assert len(f) == 4
        assert isinstance(f[0], int)
        assert isinstance(f[1], bool)
        if f[1]:
            assert isinstance(f[2], np.ndarray)
        else:
            break
        extracted_frames.append(f)

    # check if video content was read properly
    assert len(extracted_frames) == len(video_raw_frames)
    extracted_frames = np.stack([f[2] for f in extracted_frames])
    # check if pixel values differ at most by 1, codecs are lossy
    # and are allowed to slightly change the exact content of video
    assert np.max(np.abs(extracted_frames - video_raw_frames)) <= 1

    # check seek functionality
    idx = 2
    reader.change_current_frame_pointer(idx)
    # idx should appear in queue in at most `queue_size` pop_frame reads
    readed_idxs = [(await reader.pop_frame())[0] for i in range(reader_queue_size)]
    assert any([idx == i for i in readed_idxs])

    # check cleanup procedure
    num_of_tasks = len(asyncio.all_tasks())
    await reader.close()
    assert (num_of_tasks - 1) == len(asyncio.all_tasks())
