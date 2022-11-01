#!/usr/bin/python3
from operator import itemgetter
import numpy as np
import json
import fjson  # type: ignore  # fjson module adds the float_format parameter
from pathlib import Path
from typing import Awaitable, Callable, Optional
import asyncio
import time
import subprocess
from .videoReader import videoReader
from .frameLabeler import frameLabeler
import logging
from aiohttp import web
from ..utils import catch_background_task_exception
from .centerface_onnx import CenterFace

logger = logging.getLogger(__name__)


class clientComputeHandler:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        src: str | Path,
        model: Path
        | str
        | Callable[
            [np.ndarray, float], Awaitable[list[list[list[float]]]]
        ] = "backend/models/centerfaceFXdyn.onnx",
    ):
        """This class comunicates compute results over WebSocket and callbacks.

        Instance of this class sends text-type messages of 3 forms by writing directly
        to the WebSocket, the message takes form of json encoded dict with mandatory
        msg" field. If "msg" value is:
            "new file response", response to file upload by the client:
                other fields will contain key:
                    "FPS" with framerate of the video
            "lab", response with labels mapped to indexes of video frames:
                other fields will contain keys:
                    "lab" with a dict[int: list[list[float]]] of {frame_index:labels}
            "progress", notification of the approximate progress of download:
                other fields will contain keys:
                    "estimated_time_left" estimated remaining download time in seconds
                    "ratio_done" value in (0,1) range of approximate download progress

        two callbacks are provided for incoming communication:
            .serve_under_named_pipe(pipename_video: str, config: dict), request to
                generate the anonymized file
            .request_label_range(from: int, upto: int), a request for labels for
                a range of video frames

        Args:
            ws (aiohttp.web.WebSocketResponse): websocket used to send messages
            src (str | Path): path to the input video to be processed
            model: (Path | str
            | Callable[[np.ndarray, float], Awaitable[list[list[list[float]]]]]
            , optional):
                model instance or weights path that will be passed to CenterFace
                constructor. Defaults to "models/centerfaceFXdyn.onnx"
        """
        self._ws = ws
        self._src = str(src)
        if isinstance(model, str) or isinstance(model, Path):
            model = CenterFace(str(model))
        self._model = model
        self._creation_time = time.time()
        self._approx_last_usage_time = time.time()
        self._labels_to_send_queue: asyncio.Queue = asyncio.Queue(100)
        self.ok = True
        self._send_labels_runner_task = None
        self._serve_file_task: Optional[asyncio.Task] = None
        self._serial_video_reader: Optional[videoReader] = None
        self._serial_labeler: Optional[frameLabeler] = None
        self._video_reader = None
        self._labeler = None
        self._ffprobe_metadata = self._extract_metadata()

    async def close(self):
        """canceles the ._serve_file_task task that exports the post-processed video"""
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
        if self._send_labels_runner_task is not None:
            try:
                self._send_labels_runner_task.cancel()
                await self._send_labels_runner_task
            except asyncio.CancelledError:
                pass
        logger.debug("clientComputeHandler.close() done")

    async def start(self):
        """Instantiates videoReader and frameLabeler instances.
        Sends welcome message to the client, informing FPS rate of the video.
        Starts listening for commands from client.
        """
        self._video_reader = videoReader(video_src=self._src, frames_queue_size=30)
        await self._video_reader.start()
        self._labeler = frameLabeler(
            get_frame_coroutine=self._video_reader.pop_frame,
            model=self._model,
            request_different_frame_idx_callback=self._video_reader.change_current_frame_pointer,  # noqa
        )
        await self._labeler.start()
        if not self._video_reader.ok:
            logger.error("failed to setup video reader")
            self.ok = False
            return
        welcome_resp = {
            "msg": "new file response",
            "FPS": self._ffprobe_metadata["fps"],
            # "total frames": len(self._video_reader._cap),
        }
        welcome_resp = json.dumps(welcome_resp)
        await self._ws.send_str(welcome_resp)
        logger.debug(f"responded to client with: {welcome_resp}")

        self._send_labels_runner_task = asyncio.create_task(self._send_labels_runner())
        self._send_labels_runner_task.add_done_callback(catch_background_task_exception)

    def _extract_metadata(self) -> dict:
        """extracts metadata from file by using ffprobe"""
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
                self._src,
            ],
            capture_output=True,
        )
        metadata = {
            e.split("=")[0]: e.split("=")[1]
            for e in ffprobe.stdout.decode().split("\n")
            if len(e)
        }
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
        return {
            "codec_name": codec_name,
            "format_name": format_name,
            "duration": duration,
            "fps": fps,
            "pix_fmt": pix_fmt,
            "w": w,
            "h": h,
        }

    async def _serve_file_runner(self, named_pipe_path: str | Path, config: dict = {}):
        """writes post-processed, encoded video, in streaming mode (by design to a
        named pipe, but files work too). Generates data on the fly.

        Args:
            named_pipe_path (str | Path): filepath where processed, reencoded video
                should be written
            config (dict, optional): dict with keys overriding the default behaviour.
                possible values for key:
                    "treshold" is a minimum score needed to keep a bounding-box; float
                        of value in range 0-1
                    "preview-scores" is a bool; if True, the detection score will be
                        drawn for each detection
                    "shape" is one of ["rectangle", "ellipse", "bbox"]
                    "background" is one of ["blur", "pixelate", "black"].
                        This key is ignored if "shape" maps to "bbox".
                `config` Defaults to {}.
        """
        assert self._ffprobe_metadata is not None
        duration, fps, pix_fmt, codec_name, format_name, w, h = itemgetter(
            "duration", "fps", "pix_fmt", "codec_name", "format_name", "w", "h"
        )(self._ffprobe_metadata)
        logger.debug("serve_file: preparing external writer")
        writer = subprocess.Popen(
            f"ffmpeg -v error -i {self._src} -f rawvideo -pix_fmt rgb24 -s {w}x{h}"
            f" -r {fps} -i -"  # {pipename_raw_to_raw_anonymized}"
            f" -map 1:v -map 0:a? -c:v {codec_name} -f {format_name}"
            f" -pix_fmt {pix_fmt} -movflags frag_keyframe+empty_moov -"
            f" > {named_pipe_path}",
            shell=True,
            stdin=subprocess.PIPE,
            # check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert writer.stdin is not None
        logger.debug("serve_file: preparing video_reader")
        self._serial_video_reader = videoReader(
            video_src=self._src,
            frames_queue_size=30,
        )
        await self._serial_video_reader.start(
            precomputed_cfr_index_to_video_idx=getattr(
                self._video_reader, "cfr_to_vid_idx_map"
            )
            if self._video_reader
            else None
        )
        logger.debug("serve_file: preparing video_labeler")
        self._serial_labeler = frameLabeler(
            get_frame_coroutine=self._serial_video_reader.pop_frame,
            model=self._model,
        )
        if self._labeler:
            self._serial_labeler._cache = self._labeler._cache
            self._serial_labeler._cache_true_idx = self._labeler._cache_true_idx
        await self._serial_labeler.start()
        logger.debug("serve_file: entering while")
        start_time = time.time()
        frames_already_labeled = len(self._serial_labeler._cache)
        frames_left_to_be_labeled = int(duration * fps) - frames_already_labeled

        async def notify_progress(frame_idx: int):
            assert self._serial_labeler is not None
            compute_duration = time.time() - start_time
            if frames_left_to_be_labeled > 0.01 * (duration * fps):
                # the default way of estimating ratio of work done
                frames_labeled_since_export_started = 1 + (
                    len(self._serial_labeler._cache) - frames_already_labeled
                )
                approx_ratio_done = (
                    frames_labeled_since_export_started / frames_left_to_be_labeled
                )
            else:
                # if almost everything is already precomputed, the regular way would
                # return negative values which might confuse the user. Workflow lands
                # here if, for example, the user decided to change the way of
                # anonymization, e.x. user chenged shape from rectangle to ellipse
                approx_ratio_done = frame_idx / (duration * fps)
            estimated_time_left = compute_duration * (1 / approx_ratio_done - 1)
            resp = fjson.dumps(
                {
                    "msg": "progress",
                    "ratio_done": frame_idx / (duration * fps),
                    "frames_exported": frame_idx + 1,
                    "compute_duration": compute_duration,
                    "estimated_time_left": estimated_time_left,
                },
                float_format=".6f",
            )
            await self._ws.send_str(resp)

        frame_idx = 0
        last_user_notify_progress_timestamp = start_time - 1e6
        loop = asyncio.get_event_loop()
        while True:
            processed_frames = (
                await self._serial_labeler.get_next_batch_of_frames_labeled(config)
            )
            for processed_frame in processed_frames:
                if processed_frame is None:
                    logger.debug("cv2DrawingConsumer: end of data")
                    writer.stdin.close()
                    writer.terminate()  # should be terminated by now
                    await notify_progress(frame_idx)
                    await self._serial_labeler.close()
                    await self._serial_video_reader.close()
                    return
                try:
                    await loop.run_in_executor(
                        None,
                        writer.stdin.write,
                        processed_frame.astype(np.uint8).tobytes(),
                    )  # pix_fmt=rgb24
                except BrokenPipeError as e:
                    writer.stdin.close()
                    writer.terminate()
                    logger.debug(f"serve_file: writer.stdin is closed, signal: {e}")
                    await self._serial_labeler.close()
                    await self._serial_video_reader.close()
                    return
                frame_idx += 1
                if time.time() - last_user_notify_progress_timestamp > 1:
                    await notify_progress(frame_idx)
                    last_user_notify_progress_timestamp = time.time()

    async def _send_labels_runner(self):
        """manages sending the scheduled messages that contain labels.
        Should run as a background task."""
        while True:
            try:
                if self._labels_to_send_queue.qsize() > 1:
                    labels = {}
                    for i in range(self._labels_to_send_queue.qsize()):
                        labels.update(await self._labels_to_send_queue.get())
                else:
                    labels = await self._labels_to_send_queue.get()
                # resp = json.dumps(l) + "\n"
                resp = fjson.dumps(
                    {"msg": "lab", "lab": {str(k): v for k, v in labels.items()}},
                    float_format=".2f",
                )
                await self._ws.send_str(resp)
                logger.debug(f"responded with labels {min(labels)}-{max(labels)}")
            except Exception as e:
                logger.error("send_labels_runner crashed, err:", e)
                break

    async def serve_under_named_pipe(self, namedpipe: str | Path, config: dict = {}):
        """request to start generating post-processed, encoded video, in
        streaming mode (by design to a named pipe, but files work too).

        Args:
            named_pipe_path (str | Path): filepath where processed, reencoded video
                should be written
            config (dict, optional): dict with keys overriding the default behaviour.
                possible values for key:
                    "treshold" is a minimum score needed to keep a bounding-box; float
                        of value in range 0-1
                    "preview-scores" is a bool; if True, the detection score will be
                        drawn for each detection
                    "shape" is one of ["rectangle", "ellipse", "bbox"]
                    "background" is one of ["blur", "pixelate", "black"].
                        This key is ignored if "shape" maps to "bbox".
                `config` Defaults to {}.
        """
        if self._serve_file_task is not None:
            self._serve_file_task.cancel()
            try:
                await self._serve_file_task
            except asyncio.CancelledError:
                self._serve_file_task = None
                logger.debug("serve file task sucessfully cancelled")
            if self._serial_labeler is not None:
                await self._serial_labeler.close()
            if self._serial_video_reader is not None:
                await self._serial_video_reader.close()
        self._serve_file_task = asyncio.create_task(
            self._serve_file_runner(namedpipe, config)
        )
        self._serve_file_task.add_done_callback(catch_background_task_exception)

    async def request_label_range(self, beg: int, end: int, patience: int = 2):
        """request to calculate labels for frames from beg (incl.) to end (incl.).
        Labels will be sent directly over websocket, this may take some time.

        Args:
            beg (int): first frame index for which labels to return (inclusive)
            end (int): last frame index for which labels to return (inclusive)
            patience (int, optional): assumed approximate user patience for idle-ing
                in seconds. When client requests labels in some range, it usually takes
                a noticeable amount of time. For the user not to feel like everything
                hanged indefinetly, we respond with labels for smaller than requested
                range if compute takes longer than `patience`, and we will respond with
                the rest later. The actual respond time will be in the first possible
                slot after `patience` seconds, i.e. more than `patience`. Defaults to 2.
        """
        # ex. resp: {10: frame_10_label, 11: frame_11_label, ...}
        assert beg <= end
        assert self._labeler is not None
        labels = {}
        start = time.time()
        for frame_idx in range(beg, end + 1):
            labels[frame_idx] = await self._labeler.get_label(frame_idx)
            if time.time() - start > patience:
                # increase frequency of providing labels to user so the user experience
                await self._labels_to_send_queue.put(labels)
                labels = {}  # reset labels
                start = time.time()
        if labels:  # could be empty if send was set-off due to patience
            await self._labels_to_send_queue.put(labels)
