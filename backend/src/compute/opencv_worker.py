#!/usr/bin/python3
import numpy as np
import json, fjson  # fjson module adds the float_format parameter
from pathlib import Path
import asyncio
import time
import subprocess
from videoReader import videoReader
from frameLabeler import frameLabeler
import logging

logger = logging.getLogger(__name__)


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
        if self.send_labels_runner_task is not None:
            try:
                self.send_labels_runner_task.cancel()
                await self.send_labels_runner_task
            except asyncio.CancelledError:
                pass
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

