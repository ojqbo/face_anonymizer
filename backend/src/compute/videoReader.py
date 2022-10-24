import numpy as np
from pathlib import Path
import asyncio
import decord
import logging
logger = logging.getLogger(__name__)


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

    async def pop_frame(self) -> tuple[int, bool, np.ndarray]:
        """This coroutine returns the next video frame from internal buffer. It is the main interface of this class.

        Returns:
            tuple[int, bool, np.ndarray, int]: tuple containing: (index, success, frame of shape [3, H, W], true_index))
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
                raise ValueError("could not match cfr frame index to "
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
                logger.debug(f"video_reader_runner idx not in self.cfr_to_vid_idx_map, POS_FRAMES: {idx}")
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

