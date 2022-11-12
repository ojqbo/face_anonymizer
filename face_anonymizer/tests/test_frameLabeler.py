import asyncio

import numpy as np
import pytest

from face_anonymizer.src.compute.frameLabeler import frameLabeler
from face_anonymizer.tests.conftest import dummyModel, dummyReader


@pytest.mark.asyncio
async def test_frameLabeler_sequential(
    raw_frames_reader: dummyReader,
    dummy_model: dummyModel,
    video_raw_frames: np.ndarray,
):
    labeler = frameLabeler(
        get_frame_coroutine=raw_frames_reader.pop_frame,
        model=dummy_model,
        request_different_frame_idx_callback=raw_frames_reader.seek,
    )

    # check if start creates background task
    num_of_tasks = len(asyncio.all_tasks())
    await labeler.start()
    assert (num_of_tasks + 1) == len(asyncio.all_tasks())

    # check get_next_batch_of_frames_labeled() interface
    all_labeled_frames = []
    while True:
        labeled_frames = await labeler.get_next_batch_of_frames_labeled()
        for labeled_frame in labeled_frames:
            if labeled_frame is None:
                break
            else:
                assert isinstance(labeled_frame, np.ndarray)
                all_labeled_frames.append(labeled_frame)
        else:
            continue
        break

    # check if labeled frames were read properly
    assert len(all_labeled_frames) == len(video_raw_frames)
    assert np.stack(all_labeled_frames).shape == video_raw_frames.shape

    # check cleanup procedure
    num_of_tasks = len(asyncio.all_tasks())
    await labeler.close()
    assert (num_of_tasks - 1) == len(asyncio.all_tasks())


@pytest.mark.asyncio
async def test_frameLabeler_random_access(
    raw_frames_reader: dummyReader,
    dummy_model: dummyModel,
    video_raw_frames: np.ndarray,
):
    labeler = frameLabeler(
        get_frame_coroutine=raw_frames_reader.pop_frame,
        model=dummy_model,
        request_different_frame_idx_callback=raw_frames_reader.seek,
    )
    # # check if start creates background task
    num_of_tasks = len(asyncio.all_tasks())
    await labeler.start()
    assert (num_of_tasks + 1) == len(asyncio.all_tasks())

    # check random access
    consecutive_accesses = [len(video_raw_frames) - 1, 0]
    consecutive_accesses += list(range(len(video_raw_frames) // 2))
    consecutive_accesses += list(range(len(video_raw_frames) // 2))
    for i in consecutive_accesses:
        label = await labeler.get_label(i)
        # the model returns np.mean(frame)/256 as a score
        # all the frames have increasing brightness
        assert np.mean(video_raw_frames[i]) / 256 == label[0][0]

    # check cleanup procedure
    num_of_tasks = len(asyncio.all_tasks())
    await labeler.close()
    assert (num_of_tasks - 1) == len(asyncio.all_tasks())
