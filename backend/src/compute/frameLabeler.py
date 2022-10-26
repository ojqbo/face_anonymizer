from typing import Awaitable, Callable, Optional
import numpy as np
import cv2
import os
import asyncio
import time
from pathlib import Path
from .centerface_onnx import CenterFace
from ..utils import catch_background_task_exception

import logging

logger = logging.getLogger(__name__)

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hw_decoders_any;vaapi,vdpau"

class frameLabeler:
    def __init__(
        self,
        get_frame_coroutine: Awaitable[tuple[int, bool, np.ndarray]],
        model: Path | str | Callable[[np.ndarray, Optional[float]], Awaitable[list[list[float]]]] = "backend/models/centerfaceFXdyn.onnx",
        request_different_frame_idx_callback: Callable[[int], None] = None,
        batch_size: int = 4,
        batchOfLabels_queue_size: int = 2,
    ):
        """instance of this class consumes frames from videoReader, labels them, and returns batches of frames with labels
        consumed data is expected to be a tuple of (int, bool, numpy.array, int)
        designating: frame index; indicator if frame was read properly; numpy array of shape 3(RGB) x Height x Width or anything if indicator was False; true index of frame as in video file.

        Two interfaces are provided to interact with this labeler:
        get_next_batch_of_frames_labeled(config) -> list[numpy.ndarray]:
            returned data is a list of already labeled and post processed frames according to `model` instance, labelled with `_apply_labels` function, according to `config` parameter.
            If there are no more frames in video, returns None
        get_label(idx) -> list[list[float]]:
            returned data is a list of labels, labels being bounding boxes with scores of form [score, x0, y0, x1, y1]
            If there are no more frames in video, returns empty list: []

        Args:
            get_frame_coroutine (Awaitable): coroutine that returns a tuple of (index_int, is_read_successfully_bool, frame_numpy_array)
                For any `is_read_successfully_bool == False`, the `true_index` must be one of indexes already returned
                by get_frame_coroutine, preferably the last true frame index of the video.
            model (Path | str | Callable[[np.ndarray, Optional[float]], Awaitable[list[list[float]]]], optional) (Callable[int]):
                model instance or weights path that will be passed to CenterFace constructor. Defaults to "models/centerfaceFXdyn.onnx"
            request_different_frame_idx_callback (Callable[int]): coroutine that triggers video seek to a different frame index.
            batch_size (int, optional): number of frames batched and passed as input to the model. Defaults to 8.
            batchOfLabels_queue_size (int, optional): queue size for batches of frames with labels. Defaults to 2.
        """
        self._batchOfLabels_queue = asyncio.Queue(batchOfLabels_queue_size)
        self._get_frame_coroutine = get_frame_coroutine
        if isinstance(model, str) or isinstance(model, Path):
            model = CenterFace(model)
        self._model = model
        self._request_different_frame_idx_callback = (
            request_different_frame_idx_callback
        )
        self._what_is_in_queue = (
            []
        )  # list of indexes: [last_put_into_Q, ..., first_to_pop_from_Q]
        self._what_soon_will_be_in_queue = (
            []
        )  # list of frame indexes currently computed
        self._batch_size = batch_size
        self.benchmark_table = {batch_size: float("inf")}
        self._next_frame_to_read = 0  # index of frame to read
        self._cache = {
            # int: List;  frame_idx: [[score,x,y,x,y], ... ]
        }
        self._cache_true_idx = {
            # int: List;  frame_idx: [[score,x,y,x,y], ... ]
        }
        self._frames_cache = {
            # int: frame;  frame_idx: numpy_array_rgb24_anonymized_frame
        }

    async def start(self):
        """starts asyncio task that augments video frames with labels in background."""
        self._frame_labeler_runner_task = asyncio.create_task(
            self._frame_labeler_runner()
        )
        self._frame_labeler_runner_task.add_done_callback(
            catch_background_task_exception
        )

    async def _frame_labeler_runner(self):
        """this coroutine consumes from get_frame_coroutine provided at __init__,
        batches the frames, sends batches to the model instance, and puts batches of labeled frames into queue.

        produced data is a batch (list of length batch_size) of tuples of (int, bool, numpy.array, list[list[float]]),
        i.e. the consumed tuple is augmented with labels of form list[list[float]], i.e. [[0.9 0 10 0 10], [0.8 20 30 25 32]],
        designating list of labels (bounding boxes): [[score, x0, y0, x1, y1], ...] this form of label should be compatible with `_apply_label` function.
        Labels are passed directly as provided by the global `model` instance without validation.
        """
        # TODO: auto adjust batch_size based on self.benchmark_table (map of batch_size: seconds_per_sample)
        while True:
            # prepare batch to compute
            idxs: list[int] = []
            true_idxs: list[int] = []
            rets: list[bool] = []
            frames: list[np.ndarray] = []
            not_in_cache: list[bool] = []
            while len(np.unique(np.array(true_idxs)[not_in_cache])) < (self._batch_size):
                idx, ret, frame, true_idx = await self._get_frame_coroutine()
                while idx != self._next_frame_to_read:
                    logger.debug(
                        f"frameLabeler: idx != self.next_frame_to_read: {idx}!={self._next_frame_to_read}"
                    )
                    idx, ret, frame, true_idx = await self._get_frame_coroutine()
                self._what_soon_will_be_in_queue.append(idx)
                idxs.append(idx)
                true_idxs.append(true_idx)
                rets.append(ret)
                frames.append(frame)
                self._next_frame_to_read += 1
                not_in_cache.append(bool(true_idx not in self._cache_true_idx))
                if len(idxs) > 2 * self._batch_size:
                    break  # failsafe if all the labels are in cache
            # self.what_soon_will_be_in_queue = idxs
            if any([r and nc for r, nc in zip(rets, not_in_cache)]):
                to_compute = {
                    tidx:f
                    for r, nc, f, tidx in zip(rets, not_in_cache, frames, true_idxs)
                    if (r and nc)
                }
                frames_transposed = np.stack(
                    list(to_compute.values())
                ).transpose(0, 3, 1, 2)
                start = time.time()
                model_labels = await self._model(frames_transposed, 0.1)
                duration = time.time() - start
                seconds_per_sample = duration / len(frames_transposed)
                self.benchmark_table[len(frames_transposed)] = seconds_per_sample
                self._cache_true_idx.update({i: l for i, l in zip(to_compute.keys(), model_labels)})
            labels = [
                self._cache_true_idx[i]
                for i in true_idxs
            ]
            await self._batchOfLabels_queue.put((idxs, rets, frames, labels))
            self._what_is_in_queue = (
                list(reversed(self._what_soon_will_be_in_queue)) + self._what_is_in_queue
            )
            self._what_soon_will_be_in_queue = []
            # self.cache.update({i:l for i, l in zip(idxs, labels)})

    async def get_next_batch_of_frames_labeled(
        self, config: dict = {}
    ) -> list[np.ndarray | None] | None:
        """returns batched, processed frames according to config (bbox/ellipse/etc, detection scores...)
        If there are no more frames in video, returns None.
        Consecutive calls to this coroutine will return consecutive frames.

        Args:
            config (dict, optional): dictionary with keys overriding the default behaviour.
                possible values for key:
                    "treshold" is a minimum score needed to keep a bounding-box; float of value in range 0-1
                    "preview-scores" is a bool; if True, the detection score will be drawn for each detection
                    "shape" is one of ["rectangle", "ellipse", "bbox"], value "bbox" makes the next key ("background") to be ommited
                    "background" is one of ["blur", "pixelate", "black"].
                `config` Defaults to {}.

        Returns:
            list[np.ndarray | None] | None: processed frames according to config or None if no more frames in the video
        """
        idxs, rets, frames, labels = await self._batchOfLabels_queue.get()
        if not any(rets):
            return None
        self._cache.update({i: l for i, l in zip(idxs, labels)})
        result = [
            _apply_labels(f, l, config) if r else None
            for i, r, f, l in zip(idxs, rets, frames, labels)
        ]
        if len(idxs) > 0:
            self._what_is_in_queue = self._what_is_in_queue[: -len(idxs)]  # pop idxs
        return result

    async def _pop_labels(self) -> tuple[int, list[list[float]]] | None:
        """pops from internal queue and rerurns tuple (indexes, labels)

        Returns:
            tuple[int, list[list[float]]] | None: tuple of (indexes, labels) or None if no more frames in the video
        """
        idxs, rets, frames, labels = await self._batchOfLabels_queue.get()
        if len(idxs) > len(self._what_is_in_queue):
            logger.error("!! should not reach here, debug is necessary !!")
        if len(idxs) > 0:
            self._what_is_in_queue = self._what_is_in_queue[: -len(idxs)]  # pop idxs
        return idxs, labels

    async def get_label(self, idx: int) -> list[list[float]]:
        """returns list of labels, if there are no more frames in video, returns empty list: []

        Args:
            idx (int): index of frame for which labels are supposed to be returned

        Returns:
            list[list[float]]: list of bounding boxes with scores of form [score, x0, y0, x1, y1]
                score is from range 0-1,
                0<=x0<=x1<=frameWidth 0<=y0<=y1<=frameHeight. Values x0, y0, x1, y1 are floats.
        """
        if idx in self._cache:
            # label = self.cache[idx]
            return self._cache[idx]
        # ensure that label of idx is in cache (BEGIN)
        if (idx not in self._what_is_in_queue) and (
            idx not in self._what_soon_will_be_in_queue
        ):
            self._change_current_frame_pointer(idx)
            # for i in self.batchOfLabels_queue.qsize() + 1:
            #     idxs, labels = await self.pop_labels()
            #     self.cache.update({i:l for i, l in zip(idxs, labels)})
        while idx not in self._cache:
            idxs, labels = await self._pop_labels()
            self._cache.update({i: l for i, l in zip(idxs, labels)})
        return self._cache[idx]

    def _change_current_frame_pointer(self, idx: int):
        """call this method if you want to seek video

        Args:
            idx (int): frame to which to ship or rewind
        """
        self._request_different_frame_idx_callback(idx)
        self._next_frame_to_read = idx

    async def close(self):
        """canceles the task that produces data into internal queue"""
        try:
            self._frame_labeler_runner_task.cancel()
            await self._frame_labeler_runner_task
        except asyncio.CancelledError:
            pass
        logger.debug(f"frameLabeler.close() done")



def _apply_label(
    frame: np.ndarray,
    label: list[float],
    shape: str,
    background: str,
    preview_scores: bool,
) -> np.ndarray:
    """modifies the frame of shape (height, width, 3) by
    drawing bounding-box area indicated in label.

    Args:
        frame (np.ndarray): a single video frame. Gets modified in-place!
        label (list[float]): label of form [score, x0, y0, x1, y1]
        shape (str): one of ["rectangle", "ellipse", "bbox"], value "bbox" makes the next key ("background") to be ommited
        background (str): one of ["blur", "pixelate", "black"]
        preview_scores (bool): if True, the detection score will be drawn for each detection

    Returns:
        np.ndarray: modified frame
    """

    # label is in form [x,y,x,y]
    score = label[0]
    x0, y0, x1, y1 = [int(v) for v in label[1:]]
    assert (
        x0 >= 0
        and y0 >= 0
        and x0 <= x1
        and y0 <= y1
        and x1 < frame.shape[0]
        and y1 < frame.shape[1]
    ), f"bad label: {label}, frame.shape:{frame.shape}"
    center = (x1 + x0) // 2, (y1 + y0) // 2
    HW = (x1 - x0), (y1 - y0)

    roi = frame[x0:x1, y0:y1]
    if background == "black":
        roi_new = np.zeros_like(roi)
    elif background == "blur":
        pseudo_pxls = 3
        roi_new = cv2.blur(roi, (HW[0] // pseudo_pxls, HW[1] // pseudo_pxls))
    elif background == "pixelate":
        pseudo_pxls = 3
        roi_new = cv2.resize(
            roi, (pseudo_pxls, pseudo_pxls), interpolation=cv2.INTER_LINEAR
        )
        roi_new = cv2.resize(roi_new, (HW[1], HW[0]), interpolation=cv2.INTER_NEAREST)

    if shape == "ellipse":
        ellipse = np.zeros([*roi_new.shape[:2], 1])
        ellipse = cv2.ellipse(
            ellipse,
            (HW[1] // 2, HW[0] // 2),
            (HW[1] // 2, HW[0] // 2),
            0,
            0,
            360,
            1,
            -1,
        )
        frame[x0:x1, y0:y1] = (1 - ellipse) * roi + (ellipse) * roi_new
    elif shape == "rectangle":
        frame[x0:x1, y0:y1] = roi_new
    elif shape == "bbox":
        thickness = max(1, int(0.0016 * max(frame.shape)))  # 3px for 1920
        color = [10, 250, 10]
        frame = cv2.rectangle(frame, (y0, x0), (y1, x1), color, thickness)
        # roi[:thickness,:,:] = color
        # roi[-thickness:,:,:] = color
        # roi[:,:thickness,:] = color
        # roi[:,-thickness:,:] = color
        # frame[x0:x1, y0:y1] = roi

    if preview_scores:
        S = max(1, 0.001 * max(frame.shape))
        color = (10, 250, 10)
        frame = cv2.putText(
            frame, f"{score:.2f}", (y0, x0 - 1), cv2.FONT_HERSHEY_PLAIN, S, color
        )

    return frame


def _apply_labels(
    frame: np.ndarray, labels: list[list[float]], config: dict = {}
) -> np.ndarray:
    """modifies the frame of shape (height, width, 3) by
    drawing bounding-boxes indicated in labels.

    Args:
        frame (numpy.ndarray): frame to anonymize according labels and config. Gets modified in-place!
        labels (list[list[float]]): list of bounding boxes with scores of form [score, x0, y0, x1, y1]
            score is from range 0-1,
            0<=x0<=x1<=frameWidth 0<=y0<=y1<=frameHeight. Values x0, y0, x1, y1 are floats.
        config (dict, optional): dictionary with keys overriding the default behaviour.
            possible values for key:
                "treshold" is a minimum score needed to keep a bounding-box; float of value in range 0-1
                "preview-scores" is a bool; if True, the detection score will be drawn for each detection
                "shape" is one of ["rectangle", "ellipse", "bbox"], value "bbox" makes the next key ("background") to be ommited
                "background" is one of ["blur", "pixelate", "black"].
            `config` Defaults to {}.

    Returns:
        np.ndarray: modified frame
    """
    T = config.get("treshold", 0.8)
    shape = config.get("shape", "bbox")
    background = config.get("background", "black")
    preview_scores = config.get("preview-scores", True)
    assert (len(frame.shape) == 3) and (frame.shape[-1] == 3)
    assert type(T) == float
    assert shape in ["rectangle", "ellipse", "bbox"]
    assert background in ["blur", "pixelate", "black"]
    assert type(preview_scores) == bool
    # assert all([len(l)==5 for l in labels])

    # filter labels by treshold
    for l in labels:
        if l[0] >= T:
            frame = _apply_label(frame, l, shape, background, preview_scores)
    return frame
