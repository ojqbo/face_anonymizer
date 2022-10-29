import asyncio
import pytest
from pathlib import Path
import numpy as np
import subprocess


class dummyReader:
    def __init__(self, frames) -> None:
        self._idx = 0
        self._frames = frames

    async def pop_frame(self) -> tuple[int, bool, np.ndarray | list]:
        i = self._idx
        self._idx += 1
        idx_ok = i < len(self._frames)
        if idx_ok:
            f = self._frames[i]
        else:
            f = []
        true_idx = i if idx_ok else len(self._frames) - 1
        return i, idx_ok, f, true_idx

    def seek(self, idx: int) -> None:
        self._idx = idx


class dummyModel:
    def __init__(self, *args, **kwargs) -> None:
        self._lock = asyncio.Lock()

    async def __call__(
        self, batch: np.ndarray, threshold: float = 0.5
    ) -> list[list[list[float]]]:
        """labels frames [score, x0, y0, x1, y1]
        score property is equal to np.mean(frame),
        x0, y0, x1, y1 represent bbox that covers whole frame"""
        loop = asyncio.get_running_loop()
        async with self._lock:
            result = await loop.run_in_executor(None, self.forward, batch, threshold)
        return result

    def forward(
        self, batch: np.ndarray, threshold: float = 0.5
    ) -> list[list[list[float]]]:
        """labels frames [score, x0, y0, x1, y1]
        score property is equal to np.mean(frame),
        x0, y0, x1, y1 represent bbox that covers whole frame"""
        return [
            [[np.mean(f) / 256, 0, 0, f.shape[-2] - 1, f.shape[-1] - 1]] for f in batch
        ]


@pytest.fixture
def raw_frames_reader(video_raw_frames: np.ndarray) -> np.ndarray:
    return dummyReader(video_raw_frames)


@pytest.fixture
def dummy_model() -> np.ndarray:
    return dummyModel()


@pytest.fixture
def video_raw_frames() -> np.ndarray:
    videolen: int = 9
    videosiz: tuple[int, int] = (100, 200, 3)
    frames = np.stack(
        [(i * 256 / videolen) * np.ones(videosiz) for i in range(videolen)]
    )
    return frames


@pytest.fixture
def video_file(video_raw_frames: np.ndarray) -> str:
    fps = 16
    p = Path(f"backend/tests/tempfiles/testvid_fps={fps}.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    N, h, w, C = video_raw_frames.shape
    subprocess.run(
        f"ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s {w}x{h} -r {fps} -i - {p}",
        shell=True,
        input=video_raw_frames.astype(np.uint8).tobytes(),
    )
    return str(p)
