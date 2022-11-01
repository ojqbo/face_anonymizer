import asyncio
import logging
from typing import cast

import numpy as np
import onnx  # type: ignore
import onnxruntime  # type: ignore

# adapted from https://github.com/ORB-HD/deface (MIT license)


logger = logging.getLogger(__name__)


class CenterFace:
    def __init__(self, onnx_path: str = "models/centerfaceFXdyn.onnx"):
        """class providing interface to calling a model and extracting labels.

        calling instance of this class with batch of frames returns a list (batch)
        of lists of predictions. Single prediction is a list of 5 float values,
        containing score and bounding box of the detected face: [score, x0, y0, x1, y1]

        Example usage:
        >>> model = CenterFace()
        >>> batch_size = 8
        >>> batch_of_frames = np.random.randn(batch_size, 3, 1920, 1080)
        >>> labels = model(batch_of_frames)
        >>> labels_for_frame_0 = labels[0]
        >>> one_of_predictions_for_frame_0 = labels_for_frame_0[0]
        >>> score, x0, y0, x1, y1 = one_of_predictions_for_frame_0  # all floats
        >>> assert (score >= 0) and (score <= 1)
        >>> assert (x0 <= x1) and (y0 <= y1)
        >>> assert (0 <= x0) and (x0 <= 1920)
        >>> assert (0 <= x1) and (x1 <= 1920)
        >>> assert (0 <= y0) and (y0 <= 1080)
        >>> assert (0 <= y1) and (y1 <= 1080)

        Args:
            onnx_path (str, optional): path where the ONNX model is located.
                Defaults to "models/centerfaceFXdyn.onnx".
        """
        # Silence warnings about unnecessary bn initializers
        # onnxruntime.set_default_logger_severity(3)
        provider_priority_list = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        dyn_model = onnx.load(onnx_path)
        self.sess = onnxruntime.InferenceSession(
            dyn_model.SerializeToString(), providers=provider_priority_list
        )
        provider = self.sess.get_providers()[0]
        logger.info(f"ONNX running on {provider}.")
        self._lock = asyncio.Lock()

    async def __call__(
        self, batch: np.ndarray, threshold: float = 0.5
    ) -> list[list[list[float]]]:
        """executes the model and extracts detections above provided `threshold`

        Args:
            batch (np.ndarray): 4D array of shape: [batch_size, 3, height, width]
            threshold (float, optional): detection threshold, detencitons with
                score lower than threshold will be discarded. Defaults to 0.5.

        Returns:
            list[list[list[float]]]: detections for every frame in batch.
                (`batch_size` times (`N_i` times (`5` floats)))
        """
        loop = asyncio.get_running_loop()
        async with self._lock:
            result = await loop.run_in_executor(None, self.forward, batch, threshold)
        return result

    def forward(
        self, batch: np.ndarray, threshold: float = 0.5
    ) -> list[list[list[float]]]:
        """executes the model and extracts detections above provided `threshold`

        Args:
            batch (np.ndarray): 4D array of shape: [batch_size, 3, height, width]
            threshold (float, optional): detection threshold, detencitons with
                score lower than threshold will be discarded. Defaults to 0.5.

        Returns:
            list[list[list[float]]]: detections for every frame in batch.
                (`batch_size` times (`N_i` times (`5` floats)))
        """
        H, W = batch.shape[-2:]  # height and width
        S = 32
        Hpad, Vpad = S - W % S, S - H % S  # horizontal and vertical pad
        Hpad0, Hpad1 = Hpad // 2, Hpad - Hpad // 2
        Vpad0, Vpad1 = Vpad // 2, Vpad - Vpad // 2
        bigger = np.pad(batch, ((0, 0), (0, 0), (Vpad0, Vpad1), (Hpad0, Hpad1)))
        out = self.sess.run(None, {"input": bigger.astype(np.float32)})
        heatmap, scale, offset, lms = out
        labels: list[list[list[float]]] = []
        for i in range(len(heatmap)):
            dets, landmarks = self.decode(
                heatmap[i : i + 1],
                scale[i : i + 1],
                offset[i : i + 1],
                lms[i : i + 1],
                cast(tuple[int, int], bigger.shape[-2:]),
                threshold=threshold,
            )
            if len(dets) == 0:
                labels.append([])
                continue
            dets = dets[:, [4, 1, 0, 3, 2]]
            dets[:, [1, 3]] -= Vpad0
            dets[:, [2, 4]] -= Hpad0
            dets[:, :1] = dets[:, :1].round(5)
            dets[:, 1:] = dets[:, 1:].round(2)
            dets[:, 1] = dets[:, 1].clip(min=0, max=H - 1)
            dets[:, 2] = dets[:, 2].clip(min=0, max=W - 1)
            dets[:, 3] = dets[:, 3].clip(min=0, max=H - 1)
            dets[:, 4] = dets[:, 4].clip(min=0, max=W - 1)
            labels.append(dets.tolist())
        # if len(dets) > 0:
        #     dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2], dets[:, 1:4:2]
        #     lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2], lms[:, 1:10:2]
        # else:
        #     dets = np.empty(shape=[0, 5], dtype=np.float32)
        #     lms = np.empty(shape=[0, 10], dtype=np.float32)

        return labels

    def decode(
        self,
        heatmap: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray,
        landmark: np.ndarray,
        size: tuple[int, int],
        threshold: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """decodes output of the model to tuple of
        - lists of detected faces (bounding boxes) with correspondng scores,
        - and list of estimated placement of face landmarks.

        Args:
            heatmap (np.ndarray): 3D array of shape: [1, height, width], model output
            scale (np.ndarray): 3D array of shape: [2, height, width], model output
            offset (np.ndarray): 3D array of shape: [2, height, width], model output
            landmark (np.ndarray): 3D array of shape: [10, height, width], model output
            size (tuple[int, int]): `height` and `width` of a frame
            threshold (float, optional): detection threshold. Defaults to 0.1.

        Returns:
            tuple[np.ndarray, np.ndarray]: detections (bounding boxes)
                and landmarks (offsets in relation to bounding box)
        """
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = np.empty((0, 5)), np.empty((0, 10))
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = (
                    np.exp(scale0[c0[i], c1[i]]) * 4,
                    np.exp(scale1[c0[i], c1[i]]) * 4,
                )
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(
                    0, (c0[i] + o0 + 0.5) * 4 - s0 / 2
                )
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes = np.append(
                    boxes,
                    [[x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s]],
                    axis=0,
                )
                lm: list = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms = np.append(lms, [lm], axis=0)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            lms = lms[keep, :]
        return boxes, lms

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> np.ndarray:
        """non maximum suppression, removes overlapping detections
        if their intersection over union (in terms of area) is greater than `nms_tresh`

        Args:
            boxes (np.ndarray): 2D array of shape: [num_detecitons, 4]
            scores (np.ndarray): 1D array of length num_detecitons
            nms_thresh (float): threshold for IOU measure

        Returns:
            list[bool]: list of which predictions to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.uint8)
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True
        keep = np.nonzero(suppressed == 0)[0]
        return keep
