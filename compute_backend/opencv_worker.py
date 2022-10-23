#!/usr/bin/python3
from typing import Awaitable, Callable, Tuple
import numpy as np
import json, fjson  # fjson module adds the float_format parameter
from pathlib import Path
import cv2
import os
import asyncio
import time
import torch
import random
import subprocess
from centerface_onnx import CenterFace
import decord

import logging

logger = logging.getLogger(__name__)

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hw_decoders_any;vaapi,vdpau"

model = CenterFace()


class videoReader:
    """this class reads the video file provided by URI (http or file path).
    The indended usage is to initialize the instance and use coroutine pop_frame() to get frames.
    This class supports video seeking truogh function change_current_frame_pointer(int).
    videoReader will respond indefinetly, for any positive index, but the return values will be empty.
    videoReader will return frames in constant frame rate (cfr), even if the video is of 
    variable frame rate (vfr), for this reason each returned frame is accompanied with
    index (cfr) and its true_index (actual video's frame index vfr)

    Example code of reading frames 0,1,2,...4,100,101,..104
        >>> reader = videoReader(video_src="video.mp4", frames_queue_size=30)
        >>> frames = []
        >>> for _ in range(5):
        >>>     index, success, frame, true_index = await reader.pop_frame()
        >>>     frames.append((index, success, frame, true_index))
        >>> reader.change_current_frame_pointer(100)  # note: this is function, not awaitable
        >>> # pop frames that happened to be read into queue
        >>> while True:
        >>>     # this will take up to ~30 iterations (the length of queue)
        >>>     index, success, frame, true_index = await reader.pop_frame()
        >>>     if index == 100:
        >>>         break
        >>> frames.append((index, success, frame, true_index))  # append frame of index 100
        >>> # append following frames, until index 109
        >>> for _ in range(4):
        >>>     index, success, frame, true_index = await reader.pop_frame()
        >>>     frames.append((index, success, frame, true_index))
        >>> print([t[0] for t in frames])
        [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
        >>> print(type(index), type(success), type(frame))
        <class 'int'> <class 'bool'> <class 'numpy.ndarray'>

    Example showing data structure at the end of file
        >>> reader = videoReader("video.mp4")
        >>> index, success, frame, true_index = await reader.pop_frame()
        >>> print(index, success, frame.shape, true_index)
        0 True (1920, 1080, 3) 0
        >>> while success:
                last = index, success, frame, true_index
                index, success, frame, true_index = await reader.pop_frame()
        >>> print("last ok frame:", last[0], last[1], last[2].shape, last[3],
                "\\nfollowing ones:", index, success, frame, true_index)
        last ok frame: 135000 True (1920, 1080, 3) 132462
        following ones: 135001 False [] 132462
    """

    def __init__(self, video_src: str | Path, frames_queue_size: int = 10):
        """initializer of videoReader

        Args:
            video_src (str | Path): path under which the file is available. Could be http address or filepath.
            frames_queue_size (int, optional): Numbers of frames to preload to buffer. Defaults to 10.
        """
        self._frames_queue = asyncio.Queue(frames_queue_size)
        self.src = video_src
        self._what_is_in_queue = []
        self._request_new_POS_FRAMES = []
        self._next_POS_FRAMES = 0

    async def start(self, precomputed_cfr_index_to_video_idx: dict[int,int]=None) -> bool:
        """starts asyncio task reading video frames into queque in background.
        Also, if this function returned True, self.FPS property is initialized.

        Args:
            precomputed_cfr_index_to_video_idx (dict[int,int]): mapping CFR to video frame indices
                frame indices as if video was constant frame rate get mapped to their respective
                frame indices in the actual video file.

        Returns:
            bool: True if the instance was started succesfully. False otherwise, i.e. if the file could not be found under URI or file was corrupted.
        """
        loop = asyncio.get_running_loop()
        self._cap = await loop.run_in_executor(
            None, decord.VideoReader, self.src, decord.cpu(0)
        )
        self.FPS = await loop.run_in_executor(
            None,
            self._cap.get_avg_fps,
        )
        if precomputed_cfr_index_to_video_idx:
            self.cfr_to_vid_idx_map = precomputed_cfr_index_to_video_idx
        else:
            self.cfr_to_vid_idx_map = self._precompute_cfr_index_to_video_idx()
        logger.debug(f"self.cfr_to_vid_idx_map: {self.cfr_to_vid_idx_map}")
        self.ok = True
        self._video_reader_runner_task = asyncio.create_task(
            self._video_reader_runner()
        )
        return self.ok

    async def pop_frame(self) -> Tuple[int, bool, np.ndarray]:
        """This coroutine returns the next video frame from internal buffer. It is the main interface of this class.

        Returns:
            Tuple[int, bool, np.ndarray, int]: tuple containing: (index, success, frame of shape [3, H, W], true_index))
        """
        f = await self._frames_queue.get()
        self._what_is_in_queue.pop()
        return f

    def change_current_frame_pointer(self, idx: int):
        """call this method if you want to seek video to a specific frame.
        The frame for provided index is guaranteed to be eventually in the queue.
        Keep in mind that the reader can return a frame for any positive index.
        If the index is larger than number of frames in a video, say idx=int(1e6)
        then tuple (1000000, False, []) will eventually be placed into queue.

        Args:
            idx (int): absolute index of a frame to which the video reader should change the pointer to.
        """
        logger.debug(
            f"videoReader.change_current_frame_pointer({idx}) invoked, what_is_in_queue:{self._what_is_in_queue}, next_POS_FRAMES:{self._next_POS_FRAMES}"
        )
        # ensure that idx will show up in queue asap
        if (
            len(self._request_new_POS_FRAMES) > 0
            and self._request_new_POS_FRAMES[-1] == idx
        ):
            logger.debug(
                f"videoReader.change_current_frame_pointer({idx}): already scheduled, just wait"
            )
            return  # already scheduled, just wait
        if idx in self._what_is_in_queue:
            logger.debug(
                f"videoReader.change_current_frame_pointer({idx}): {idx} is in queue, just read it"
            )
            return
        dont_seek_if_X_frames_ahead = 2  # this value must be >= 2
        if (idx >= self._next_POS_FRAMES) and (
            idx <= self._next_POS_FRAMES + dont_seek_if_X_frames_ahead
        ):
            logger.debug(
                f"videoReader.change_current_frame_pointer({idx}): {idx} will soon be read, just continue reading"
            )
            return
        # should set, schedule seek:
        logger.debug(
            f"videoReader.change_current_frame_pointer({idx}): [{idx}] + {self._request_new_POS_FRAMES}"
        )
        self._request_new_POS_FRAMES = [idx] + self._request_new_POS_FRAMES

    def _precompute_cfr_index_to_video_idx(self):
        cfr_to_vid_idx_map = {}
        cfr_idx = 0  # constant frame rate idx
        vid_idx = 0  # video indices (either variable or constant frame rate)
        num_of_video_frames = len(self._cap)
        while vid_idx < num_of_video_frames:
            timestamp = cfr_idx / self.FPS
            s, e = self._cap.get_frame_timestamp(vid_idx)
            if (timestamp >= s) and (timestamp < e):
                cfr_to_vid_idx_map[cfr_idx] = vid_idx
                cfr_idx += 1
            elif (timestamp < s):
                cfr_idx += 1
            elif (timestamp >= e):
                vid_idx += 1
            else:
                raise ValueError("could not match cfr frame index to"
                "video frame index, code should not reach here")
        # with this above alone, the resulting video looks a bit choppy,
        #  lets add some tolerance if we've got near miss resulting in 2-frames skipped
        cfr_idx = 1
        while cfr_idx + 1 in cfr_to_vid_idx_map:
            a, b = cfr_to_vid_idx_map[cfr_idx-1], cfr_to_vid_idx_map[cfr_idx+1]
            if (b - a) > 1:
                timestamp = ((b + a) // 2) / self.FPS
                t_a = self._cap.get_frame_timestamp(a)[1]
                t_b = self._cap.get_frame_timestamp(b)[0]
                t_mean = (t_b + t_a) / 2
                dt_acceptable = (1 / self.FPS) * 1.15  # 15% tolerance
                if np.abs(timestamp - t_mean) <= dt_acceptable / 2:
                    cfr_to_vid_idx_map[cfr_idx] = (b + a) // 2
            cfr_idx += 1
        return cfr_to_vid_idx_map

    async def _video_reader_runner(self):
        """This coroutine reads the video and puts consecutive frames into queue. This task is ment to be run in background.
        It is this task that fills `self._frames_queue` with tuples of `(index, success_status, frame, true_index)`.
        Note that it reads the video indefinetly, there is no stop condition. It is assumed that down the pipeline,
        some consumer of this class instance data will detect that this instance returns empty frames
        (index>=frames_in_video, success_status==False, frame==[]) and no further requests will be done.
        This leads the queue to be fiilled with empty frames at the end of the video, this is intentional."""
        loop = asyncio.get_running_loop()
        idx = 0
        last_iteration_idx_in_video = -1
        self._next_POS_FRAMES = idx
        logger.debug(f"video_reader_runner started, POS_FRAMES: {idx}")
        while True:
            if self._request_new_POS_FRAMES:
                idx = self._request_new_POS_FRAMES.pop()
                logger.debug(
                    f"change in current frame index detected, new POS_FRAMES: {idx}"
                )
            if idx in self.cfr_to_vid_idx_map:
                idx_in_video = self.cfr_to_vid_idx_map[idx]
                ret = True
            else:
                idx_in_video = len(self._cap)-1
                ret = False 
                logger.debug(f"video_reader_runner idx in self.cfr_to_vid_idx_map, POS_FRAMES: {idx}")
            if last_iteration_idx_in_video != idx_in_video:
                last_iteration_idx_in_video = idx_in_video
                await loop.run_in_executor(
                    None,
                    self._cap.seek_accurate,
                    idx_in_video,
                )
                try:
                    frame = await loop.run_in_executor(None, self._cap.next)
                    frame = frame.asnumpy()
                    ret = True
                except StopIteration:
                    ret = False
            self._next_POS_FRAMES = idx + 1
            await self._frames_queue.put(
                (idx, ret, frame if ret else [], idx_in_video)
            )
            logger.debug(
                f"videoReader.video_reader_runner: ret:{ret} updating what_is_in_queue: [{idx}] + {self._what_is_in_queue}"
            )
            self._what_is_in_queue = [idx] + self._what_is_in_queue  # .insert(0, idx)
            idx = idx + 1

    async def close(self):
        """closes the reader and cancelles the task that puts frames into queue"""
        try:
            self._video_reader_runner_task.cancel()
            await self._video_reader_runner_task
        except asyncio.CancelledError:
            pass
        del self._cap
        logger.debug(f"videoReader.close() done")


class frameLabeler:
    def __init__(
        self,
        get_frame_coroutine: Awaitable[tuple[int, bool, np.ndarray]],
        request_different_frame_idx_callback: Callable[[int], None] = None,
        batch_size: int = 8,
        batchOfLabels_queue_size: int = 2,
    ):
        """instance of this class consumes frames from videoReader, labels them, and returns batches of frames with labels
        consumed data is expected to be a tuple of (int, bool, numpy.array, int)
        designating: frame index; indicator if frame was read properly; numpy array of shape 3(RGB) x Height x Width or anything if indicator was False; true index of frame as in video file.

        Two interfaces are provided to interact with this labeler:
        get_next_batch_of_frames_labeled(config) -> list[numpy.ndarray]:
            returned data is a list of already labeled and post processed frames according to `model` instance, labelled with `apply_labels` function, according to `config` parameter.
            If there are no more frames in video, returns None
        get_label(idx) -> list[list[float]]:
            returned data is a list of labels, labels being bounding boxes with scores of form [score, x0, y0, x1, y1]
            If there are no more frames in video, returns empty list: []

        Args:
            get_frame_coroutine (Awaitable): coroutine that returns a tuple of (index_int, is_read_successfully_bool, frame_numpy_array)
            request_different_frame_idx_callback (Callable[int]): coroutine that triggers video seek to a different frame index.
            batch_size (int, optional): number of frames batched and passed as input to the model. Defaults to 8.
            batchOfLabels_queue_size (int, optional): queue size for batches of frames with labels. Defaults to 2.
        """
        self._batchOfLabels_queue = asyncio.Queue(batchOfLabels_queue_size)
        self._get_frame_coroutine = get_frame_coroutine
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

    def change_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def get_batch_size(self) -> int:
        return self._batch_size

    async def start(self):
        """starts asyncio task that augments video frames with labels in background."""
        self.frame_labeler_runner_task = asyncio.create_task(
            self._frame_labeler_runner()
        )

    async def _frame_labeler_runner(self):
        """this coroutine consumes from get_frame_coroutine provided at __init__,
        batches the frames, sends batches to the model instance, and puts batches of labeled frames into queue.

        produced data is a batch (list of length batch_size) of tuples of (int, bool, numpy.array, list[list[float]]),
        i.e. the consumed tuple is augmented with labels of form list[list[float]], i.e. [[0.9 0 10 0 10], [0.8 20 30 25 32]],
        designating list of labels (bounding boxes): [[score, x0, y0, x1, y1], ...] this form of label should be compatible with `apply_label` function.
        Labels are passed directly as provided by the global `model` instance without validation.
        """
        # TODO: auto adjust batch_size based on self.benchmark_table (map of batch_size: seconds_per_sample)
        loop = asyncio.get_running_loop()
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
                logger.debug(
                    f"idx == self.next_frame_to_read: {idx}=={self._next_frame_to_read}; ret:{ret}"
                )
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
                logger.debug(f"frameLabeler: any(rets) is True, sum(rets): {sum(rets)}")
                to_compute = {
                    tidx:f
                    for r, nc, f, tidx in zip(rets, not_in_cache, frames, true_idxs)
                    if (r and nc)
                }
                frames_transposed = np.stack(
                    list(to_compute.values())
                ).transpose(0, 3, 1, 2)
                start = time.time()
                model_labels = await loop.run_in_executor(
                    None, model, frames_transposed, 0.1
                )
                duration = time.time() - start
                seconds_per_sample = duration / len(frames_transposed)
                self.benchmark_table[len(frames_transposed)] = seconds_per_sample
                self._cache_true_idx.update({i: l for i, l in zip(to_compute.keys(), model_labels)})
            labels = [
                self._cache_true_idx[i]
                for i in true_idxs
            ]
            logger.debug(
                f"frameLabeler: await queue.put(idxs,len(labels)): {(idxs, len(labels))}"
            )
            await self._batchOfLabels_queue.put((idxs, rets, frames, labels))
            self._what_is_in_queue = (
                list(reversed(self._what_soon_will_be_in_queue)) + self._what_is_in_queue
            )
            logger.debug(
                f"frameLabeler: self.what_is_in_queue: {self._what_is_in_queue}"
            )
            self._what_soon_will_be_in_queue = []
            # self.cache.update({i:l for i, l in zip(idxs, labels)})

    async def get_next_batch_of_frames_labeled(
        self, config: dict = {}
    ) -> list[np.ndarray] | None:
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
            list[np.ndarray] | None: processed frames according to config or None if no more frames in the video
        """
        logger.debug(f"get_next_batch_of_frames_labeled invoked")
        idxs, rets, frames, labels = await self._batchOfLabels_queue.get()
        if not any(rets):
            return None
        logger.debug(f"frameLabeler.get_labeled_frame: got (idxs): {(idxs)}")
        result = [
            apply_labels(f, l, config) if r else None
            for i, r, f, l in zip(idxs, rets, frames, labels)
        ]
        self._cache.update({i: l for i, l in zip(idxs, labels)})
        if len(idxs) > 0:
            self._what_is_in_queue = self._what_is_in_queue[: -len(idxs)]  # pop idxs
        logger.debug(
            f"get_next_batch_of_frames_labeled done, len(result): {len(result)}"
        )
        return result

    async def _pop_labels(self) -> tuple[int, list[list[float]]] | None:
        """pops from internal queue and rerurns tuple (indexes, labels)

        Returns:
            tuple[int, list[list[float]]] | None: tuple of (indexes, labels) or None if no more frames in the video
        """
        idxs, rets, frames, labels = await self._batchOfLabels_queue.get()
        logger.debug(f"frameLabeler.pop_labels: got (idxs,len(labels)): {(idxs, len(labels))}")
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
        logger.debug(f"frameLabeler.get_label({idx}) invoked")
        if idx in self._cache:
            logger.debug(f"frameLabeler.get_label({idx}) {idx} was in cache.")
            # label = self.cache[idx]
            return self._cache[idx]
        # ensure that label of idx is in cache (BEGIN)
        if (idx not in self._what_is_in_queue) and (
            idx not in self._what_soon_will_be_in_queue
        ):
            logger.debug(
                f"frameLabeler.get_label({idx}) not in none of: cache, queue, soon_in_queue"
            )
            self._change_current_frame_pointer(idx)
            # for i in self.batchOfLabels_queue.qsize() + 1:
            #     idxs, labels = await self.pop_labels()
            #     self.cache.update({i:l for i, l in zip(idxs, labels)})
        while idx not in self._cache:
            logger.debug(f"frameLabeler.get_label({idx}) calling pop_labels...")
            idxs, labels = await self._pop_labels()
            self._cache.update({i: l for i, l in zip(idxs, labels)})
        logger.debug(
            f"frameLabeler.get_label({idx}) {idx} finally in cache, returning."
        )
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
            self.frame_labeler_runner_task.cancel()
            await self.frame_labeler_runner_task
        except asyncio.CancelledError:
            pass
        logger.debug(f"frameLabeler.close() done")


class clientComputeHandler:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """instance of this class comunicates by asyncio.StreamReader and asyncio.StreamWriter (a TCP socket).
        No public functions or properties are present in this class.

        Expected usage:
            >>> H = clientComputeHandler(reader, writer)
            >>> await H.start()  # ends at connection close (EOF from reader)
            >>> await H.close()
            >>> writer.close()
            >>> del H

        reader/writer API (API for TCP communication):
        each incoming and outgoing message is expected to end with newline b"\n" character

        reader
        accepts two message types:
            "serve under named_pipe: " + pipe_name
                where pipe_name is the file to which exported video is supposed to be written
            str(from_int) + "-" + str(upto_int)
                a request for labels corresponding to all frames from from_int ip to uoto_int (inclusive)
                this message initiate preparing and eventual response with b"labels" message to the writer

        writer
        sends asynchronously only two types of messages:
            b"labels" + payload + b"\n"
                where payload is .encode()'d json string of form dict[str,list[list[float]]],
                i.e. {"0":[], "1":[[0.99, 0.00, 10.00, 5.00, 15.00]]}
            b"progress" + payload + b"\n"
                where payload is .encode()'d json string of dict:
                {
                    "frames_exported": frame_idx + 1,
                    "duration": time.time() - started_exporting_timestamp
                }
        one exception is the initial message of form:
            payload + b"\n"
                payload containing .encode()'d json string of dict {"FPS":average_video_fps}
                this is because FPS field is used on the front-end side, as browsers currently
                have no real interface to extract this property locally.

        Args:
            reader (asyncio.StreamReader): reader of client messages
            writer (asyncio.StreamWriter): writer of messages to the client
        """
        self._creation_time = time.time()
        self._approx_last_usage_time = time.time()
        self._reader, self._writer = reader, writer
        self._client = self._writer.get_extra_info("peername")
        self._labels_to_send_queue = asyncio.Queue(100)
        self.ok = True
        self._serve_file_task = None
        self._serial_video_reader = None
        self._video_reader = None
        self._labeler = None
        self._serial_labeler = None

    async def close(self):
        """canceles the `self._serve_file_task` task that exports the post-processed video"""
        if self._serve_file_task is not None:
            try:
                self._serve_file_task.cancel()
                await self._serve_file_task
            except asyncio.CancelledError:
                pass
        if self._serial_video_reader is not None:
            await self._serial_video_reader.close()
        if self._video_reader is not None:
            await self._video_reader.close()
        if self._serial_labeler is not None:
            await self._serial_labeler.close()
        if self._labeler is not None:
            await self._labeler.close()
        logger.debug(f"clientComputeHandler.close() handler for client {self._client}")

    async def start(self):
        """Instantiates videoReader and frameLabeler instances.
        Sends welcome message to the client, informing FPS rate of the video.
        Starts listening for commands from client.
        """
        setup_msg = await self._reader.readline()
        setup_msg = setup_msg.decode()
        logger.debug(f"recieved init message: {setup_msg} from {self._client}")
        setup_msg = json.loads(setup_msg)
        if (
            self._client[0] in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
            and "path" in setup_msg
        ):
            # if server is on localhost and provides a filepath
            src = setup_msg["path"]
        else:
            src = setup_msg["src"].replace("APP_HOST", self._client[0], 1)
        self.src = src
        self.filename = Path(src).name
        self._video_reader = videoReader(video_src=self.src, frames_queue_size=30)
        await self._video_reader.start()
        self._labeler = frameLabeler(
            get_frame_coroutine=self._video_reader.pop_frame,
            request_different_frame_idx_callback=self._video_reader.change_current_frame_pointer,
            batch_size=8,
            batchOfLabels_queue_size=2,
        )
        await self._labeler.start()
        if not self._video_reader.ok:
            logger.error(f"failed to setup video reader")
            self.ok = False
            return
        welcome_resp = {
            "FPS": self._video_reader.FPS,  # self.video_reader.cap.get(cv2.CAP_PROP_FPS),
            # "total frames": int(self.video_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        welcome_resp = (json.dumps(welcome_resp) + "\n").encode()
        self._writer.write(welcome_resp)
        await self._writer.drain()
        logger.debug(f"responded to client {self._client} with: {welcome_resp}")

        self.send_labels_runner_task = asyncio.create_task(self._send_labels_runner())
        await self.start_communicating()

    async def _serve_file_runner(self, named_pipe_path: str | Path, config: dict = {}):
        """writes post-processed, encoded video, in streaming mode (generated on the fly).

        Args:
            named_pipe_path (str | Path): filepath where processed, reencoded video should be written
            config (dict, optional): dictionary with keys overriding the default behaviour.
                possible values for key:
                    "treshold" is a minimum score needed to keep a bounding-box; float of value in range 0-1
                    "preview-scores" is a bool; if True, the detection score will be drawn for each detection
                    "shape" is one of ["rectangle", "ellipse", "bbox"], value "bbox" makes the next key ("background") to be ommited
                    "background" is one of ["blur", "pixelate", "black"].
                `config` Defaults to {}.
        """
        ffprobe = subprocess.run(
            [
                "ffprobe",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_tag,codec_name,width,height,avg_frame_rate,pix_fmt",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "default=noprint_wrappers=1",
                self.src,
            ],
            capture_output=True,
        )
        metadata = ffprobe.stdout.decode().split("\n")
        metadata = {e.split("=")[0]: e.split("=")[1] for e in metadata if len(e)}
        logger.info(f"ffprobe: {metadata}")
        w, h = metadata["width"], metadata["height"]
        pix_fmt = metadata["pix_fmt"]
        fps = int(metadata["avg_frame_rate"].split("/")[0]) / int(
            metadata["avg_frame_rate"].split("/")[1]
        )
        codec_name = metadata["codec_name"]
        format_name = metadata["format_name"]
        duration = float(metadata["duration"])
        if len(format_name.split(",")):
            format_name = format_name.split(",")[0]
        logger.debug(f"serve_file: preparing external writer")
        writer = subprocess.Popen(
            f"ffmpeg -v error -i {self.src} -f rawvideo -pix_fmt rgb24 -s {w}x{h}"
            f" -r {fps} -i -"  # {pipename_raw_to_raw_anonymized}"
            f" -map 1:v -map 0:a? -c:v {codec_name} -f {format_name}"
            f" -pix_fmt {pix_fmt} -movflags frag_keyframe+empty_moov - > {named_pipe_path}",
            shell=True,
            stdin=subprocess.PIPE,
            # check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.debug(f"serve_file: preparing video_reader")
        self._serial_video_reader = videoReader(
            video_src=self.src,
            frames_queue_size=30,
        )
        await self._serial_video_reader.start(
            precomputed_cfr_index_to_video_idx=self._video_reader.cfr_to_vid_idx_map
        )
        logger.debug(f"serve_file: preparing video_labeler")
        self._serial_labeler = frameLabeler(
            get_frame_coroutine=self._serial_video_reader.pop_frame,
            batch_size=8,
            batchOfLabels_queue_size=2,
        )
        # todo add syncing cache of labelers
        self._serial_labeler._cache = self._labeler._cache  # syncing like this?
        self._serial_labeler._cache_true_idx = self._labeler._cache_true_idx  # syncing like this?
        await self._serial_labeler.start()
        logger.debug(f"serve_file: entering while")
        start_time = time.time()

        async def notify_progress(frame_idx: int):
            approx_ratio_done = frame_idx / (duration * fps)
            compute_duration = time.time() - start_time
            estimated_time_left = compute_duration * (1 / approx_ratio_done - 1)
            resp = (
                "progress"
                + fjson.dumps(
                    {
                        "ratio_done": approx_ratio_done,
                        "frames_exported": frame_idx + 1,
                        "compute_duration": compute_duration,
                        "estimated_time_left": estimated_time_left,
                    },
                    float_format=".6f",
                )
                + "\n"
            )
            self._writer.write(resp.encode())

        frame_idx = 0
        last_user_notify_progress_timestamp = start_time - 1e6
        while True:
            processed_frames = (
                await self._serial_labeler.get_next_batch_of_frames_labeled(config)
            )
            for processed_frame in processed_frames:
                if processed_frame is None:
                    logger.debug(f"cv2DrawingConsumer: end of data")
                    writer.stdin.close()
                    writer.terminate()  # should be terminated by now
                    await notify_progress(frame_idx)
                    return
                processed_frame = processed_frame.astype(np.uint8).tobytes()
                logger.debug(
                    f"serve_file: new_frame ready, writing {len(processed_frame)} bytes to writer.stdin"
                )
                writer.stdin.write(processed_frame)
                # await loop.run_in_executor(
                #     None,
                #     writer.stdin.buffer.write,
                #     processed_frame
                # ) # pix_fmt=rgb24
                frame_idx += 1
                if time.time() - last_user_notify_progress_timestamp > 1:
                    await notify_progress(frame_idx)
                    last_user_notify_progress_timestamp = time.time()

    async def _send_labels_runner(self):
        """manages sending the scheduled messages that contain labels. Background runner."""
        while True:
            try:
                if self._labels_to_send_queue.qsize() > 1:
                    l = {}
                    for i in range(self._labels_to_send_queue.qsize()):
                        l.update(await self._labels_to_send_queue.get())
                else:
                    l = await self._labels_to_send_queue.get()
                # resp = json.dumps(l) + "\n"
                resp = (
                    "labels"
                    + fjson.dumps({str(k): v for k, v in l.items()}, float_format=".2f")
                    + "\n"
                )
                self._writer.write(resp.encode())
                await self._writer.drain()
                logger.debug(f"responded with labels {min(l)}-{max(l)}")
            except Exception as e:
                logger.error("send_labels_runner crashed, err:", e)
                # break?

    async def start_communicating(self, patience: int = 2):
        """listens to messages from client and responds to those messages

        Args:
            patience (int, optional): assumed approximate patience of the user in seconds.
                When client requests labels in some range, it usually takes a noticeable amount of time.
                For the user not to feel like everything hanged indefinetly, we respond with labels for smaller
                than requested range if compute tales longer than `patience` and will respond with the rest later.
                The actual respond time will be in the first possible slot after `patience` seconds,
                i.e. more than `patience`. Defaults to 2.
        """
        corrupted_msg_counter = 0
        # loop = asyncio.get_running_loop()
        while True:
            # await asyncio.gather(*[t for t in tasks if t.done()])
            # tasks = [t for t in tasks if not t.done()] # remove references to finished tasks
            self._approx_last_usage_time = time.time()  # variable unused for now
            line = await self._reader.readline()
            if len(line) == 0:
                corrupted_msg_counter += 1
                if self._reader.at_eof():
                    return
            if corrupted_msg_counter > 10:
                # should never get here
                self._writer.write(
                    "recieved too many corrupted msgs, disconnecting\n".encode()
                )
                await self._writer.drain()
                logger.debug(
                    f"got too many corrupted msgs ({corrupted_msg_counter}) aborting requests_reader_runner"
                )
                return
            logger.debug(f"command from {self._client} arrived: {line}")
            command = line.decode().rstrip()
            if command.startswith("serve under named_pipe: "):
                msg = json.loads(command.removeprefix("serve under named_pipe: "))
                if self._serve_file_task is not None:
                    self._serve_file_task.cancel()
                    try:
                        await self._serve_file_task
                    except asyncio.CancelledError:
                        logger.debug("serve file task sucessfully cancelled")
                self._serve_file_task = asyncio.create_task(
                    self._serve_file_runner(msg["namedpipe"], msg["config"])
                )
                # todo change to writer.write
                # with open(f"labels_cache_{interactive_opt_filename}.json","w") as f:
                #     json.dump(interactive_cache, f)
                continue
            elif command.find("-") != -1:
                # command for range of frames
                # ex. command: 10-25, resp: {10: frame_10_label, 11: frame_11_label, ...}
                beg, end = [int(x) for x in command.split("-")]
                assert beg < end
                # labels = {frame_idx: self.label_of_frame(frame_idx) for frame_idx in range(beg, end)}
                labels = {}
                start = time.time()
                for frame_idx in range(beg, end + 1):
                    labels[frame_idx] = await self._labeler.get_label(frame_idx)
                    # labels[frame_idx] = await self.label_of_frame(frame_idx, batch_size)
                    if time.time() - start > patience:
                        # increase frequency of providing labels to user so the user experience
                        await self._labels_to_send_queue.put(labels)
                        # await self.send_labels(labels)
                        # tasks.append(asyncio.create_task(self.send_labels(labels)))
                        labels = {}  # reset labels
                        start = time.time()
                if labels:  # could be empty if send was set-off due to patience
                    await self._labels_to_send_queue.put(labels)
                    # await self.send_labels(labels)
                    # tasks.append(asyncio.create_task(self.send_labels(labels)))
            else:
                corrupted_msg_counter += 1
                continue
                # command=int(command)
                # labels = {command: self.label_of_frame(command)}#this is a single label, despite the labels variable name


def apply_label(
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


def apply_labels(
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
    assert type(T) == float
    assert shape in ["rectangle", "ellipse", "bbox"]
    assert background in ["blur", "pixelate", "black"]
    assert type(preview_scores) == bool
    # assert all([len(l)==5 for l in labels])

    # filter labels by treshold
    for l in labels:
        if l[0] >= T:
            frame = apply_label(frame, l, shape, background, preview_scores)
    return frame
