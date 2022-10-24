import asyncio
from re import sub
import pytest
from pathlib import Path
import numpy as np
import subprocess
from backend.src.compute.videoReader import videoReader

@pytest.fixture
def video_raw_frames() -> np.ndarray:
    videolen: int = 8
    videosiz: tuple[int, int] = (100, 200, 3)
    frames = np.stack([
        (i*256/videolen)*np.ones(videosiz, dtype=np.uint8)
        for i in range(videolen)
    ])
    return frames

@pytest.fixture
def video_file(video_raw_frames: np.ndarray) -> str:
    p = Path("backend/test/tempfiles/testvid.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    N, h, w, C = video_raw_frames.shape
    subprocess.run(
        f"ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s {w}x{h} -r 16 -i - {p}",
        shell=True, input=video_raw_frames.astype(np.uint8).tobytes()
    )
    return str(p)

@pytest.mark.asyncio
async def test_videoReader(video_raw_frames: np.ndarray, video_file: str):
    reader = videoReader(video_src=video_file, frames_queue_size=10)
    await reader.start()
    extracted_frames = []
    while True:
        f = await reader.pop_frame()
        assert isinstance(f[0],int)
        assert isinstance(f[1],bool)
        if f[1]:
            assert isinstance(f[2],np.ndarray)
        if not f[1]:
            break
        extracted_frames.append(f)
    assert len(extracted_frames) == len(video_raw_frames)
    extracted_frames = np.stack([f[2] for f in extracted_frames])
    assert np.mean(extracted_frames - video_raw_frames) < 1e-1
