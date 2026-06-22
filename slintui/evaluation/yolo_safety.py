"""Safety YOLO 检测器 (workwear / breakerON / breakerOFF)，基于 RKNN NPU 推理

与 evaluation yolo.py 相同的 DFL 输出结构，支持 batch 推理（两路摄像头同时推理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
import time
from numba import njit, prange
from rknnlite.api import RKNNLite as RKNN

IMG_SIZE = (640, 640)
CLASSES = ("workwear", "breakerOFF", "breakerON", "person")
OBJ_THRESH = 0.25
NMS_THRESH = 0.6


@dataclass
class SafetyBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float


@dataclass
class SafetyResult:
    boxes: List[SafetyBox]


@njit(fastmath=True)
def _dfl(position):
    n, c = position.shape
    p_num = 4
    mc = c // p_num
    out = np.empty((n, 4), dtype=np.float32)

    for i in prange(n):
        for p in range(4):
            base = p * mc
            max_val = -1e9
            for k in range(mc):
                if position[i, base + k] > max_val:
                    max_val = position[i, base + k]

            s = 0.0
            for k in range(mc):
                s += np.exp(position[i, base + k] - max_val)

            acc = 0.0
            for k in range(mc):
                acc += np.exp(position[i, base + k] - max_val) * k
            out[i, p] = acc / s
    return out


def _process_branch(box_in, cls_in):
    n, c, h, w = box_in.shape
    box_raw = box_in.transpose(0, 2, 3, 1).reshape(n, -1, c)
    cls_score = cls_in.transpose(0, 2, 3, 1).reshape(n, -1, cls_in.shape[1])

    col = np.arange(w)
    row = np.arange(h)
    col, row = np.meshgrid(col, row)
    grid = np.stack((col, row), axis=-1).reshape(1, -1, 2).astype(np.float32)

    stride = 640 / h
    return box_raw, cls_score, grid, stride


class YoloSafety:
    CLASSES = ("workwear", "breakerOFF", "breakerON")
    OBJ_THRESH = 0.25
    NMS_THRESH = 0.6

    def __init__(self):
        self.latest_result: SafetyResult = SafetyResult(boxes=[])
        model_path = "./batch2-rkfork-electric-safetyv107.rknn"
        self.rknn = RKNN()
        print(f"[YoloSafety] Loading RKNN model: {model_path}")
        if self.rknn.load_rknn(model_path) != 0:
            print("[YoloSafety] Load RKNN model failed")
            self.rknn = None
            return
        if self.rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2) != 0:
            print("[YoloSafety] Init runtime environment failed")
            self.rknn = None
            return
        print("[YoloSafety] Model loaded successfully")

    def pre_process(self, bgr: np.ndarray):
        """全图 letterbox 缩放到 640x640，返回单图 RGB uint8 + 元信息 + 原始尺寸"""
        orig_h, orig_w = bgr.shape[:2]
        scale = min(IMG_SIZE[0] / orig_w, IMG_SIZE[1] / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = IMG_SIZE[0] - new_w
        pad_h = IMG_SIZE[1] - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        if pad_w > 0 or pad_h > 0:
            letterbox = cv2.copyMakeBorder(
                resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
        else:
            letterbox = resized

        rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)

        meta = {
            "ratio": scale,
            "pad": (float(pad_left), float(pad_top)),
        }
        return rgb, meta, (orig_h, orig_w)

    def post_process_batch(self, outputs, metas_list, orig_shapes):
        """DFL + NMS，支持 batch 维度；每张图独立 NMS，分别输出"""
        if outputs is None or len(outputs) == 0:
            return None, None, None

        batch_size = len(orig_shapes)

        results = [_process_branch(outputs[i * 3], outputs[i * 3 + 1]) for i in range(3)]

        boxes_raw_cat = np.concatenate([r[0] for r in results], axis=1)
        scores_cat = np.concatenate([r[1] for r in results], axis=1)
        grids_cat = np.concatenate([np.tile(r[2], (batch_size, 1, 1)) for r in results], axis=1)
        strides_cat = np.concatenate(
            [np.full((batch_size, r[0].shape[1], 1), r[3], dtype=np.float32) for r in results], axis=1
        )

        all_boxes = []
        all_classes = []
        all_scores = []

        for b in range(batch_size):
            orig_h, orig_w = orig_shapes[b]
            meta = metas_list[b]
            ratio = meta["ratio"]
            pad_x, pad_y = meta["pad"]

            b_scores = scores_cat[b]
            class_ids = np.argmax(b_scores, axis=1)
            max_scores = np.max(b_scores, axis=1)
            mask = max_scores >= self.OBJ_THRESH

            if not np.any(mask):
                all_boxes.append(np.empty((0, 4), dtype=np.float32))
                all_classes.append(np.empty((0,), dtype=np.int32))
                all_scores.append(np.empty((0,), dtype=np.float32))
                continue

            f_box_raw = boxes_raw_cat[b][mask]
            f_grid = grids_cat[b][mask]
            f_stride = strides_cat[b][mask]
            f_score = max_scores[mask]
            f_class = class_ids[mask]

            reg = _dfl(f_box_raw)
            lt = reg[:, 0:2]
            rb = reg[:, 2:4]
            f_grid_expanded = f_grid + 0.5
            x1y1 = (f_grid_expanded - lt) * f_stride
            x2y2 = (f_grid_expanded + rb) * f_stride
            f_box = np.concatenate((x1y1, x2y2), axis=1)

            f_box[:, 0::2] = (f_box[:, 0::2] - pad_x) / ratio
            f_box[:, 1::2] = (f_box[:, 1::2] - pad_y) / ratio

            np.clip(f_box[:, 0], 0, orig_w, out=f_box[:, 0])
            np.clip(f_box[:, 1], 0, orig_h, out=f_box[:, 1])
            np.clip(f_box[:, 2], 0, orig_w, out=f_box[:, 2])
            np.clip(f_box[:, 3], 0, orig_h, out=f_box[:, 3])

            boxes_wh = f_box.copy()
            boxes_wh[:, 2] -= boxes_wh[:, 0]
            boxes_wh[:, 3] -= boxes_wh[:, 1]
            indices = cv2.dnn.NMSBoxesBatched(
                boxes_wh, f_score, f_class.astype(np.int32), self.OBJ_THRESH, self.NMS_THRESH
            )

            if len(indices) == 0:
                all_boxes.append(np.empty((0, 4), dtype=np.float32))
                all_classes.append(np.empty((0,), dtype=np.int32))
                all_scores.append(np.empty((0,), dtype=np.float32))
                continue

            indices = indices.flatten()
            all_boxes.append(f_box[indices])
            all_classes.append(f_class[indices])
            all_scores.append(f_score[indices])

        return all_boxes, all_classes, all_scores

    def detect_batch(self, bgr_list: list[np.ndarray]) -> list[SafetyResult]:
        """批量推理：一次 RKNN inference 同时处理多张图（batch2 模型）"""
        if self.rknn is None:
            return [SafetyResult(boxes=[]) for _ in bgr_list]

        t_pre_start = time.perf_counter()
        imgs, metas_list, orig_shapes = [], [], []
        for bgr in bgr_list:
            rgb, meta, shape = self.pre_process(bgr)
            imgs.append(rgb)
            metas_list.append(meta)
            orig_shapes.append(shape)
        input_data = np.ascontiguousarray(np.stack(imgs, axis=0), dtype=np.uint8)
        t_pre_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()
        boxes_list, classes_list, scores_list = self.post_process_batch(outputs, metas_list, orig_shapes)
        t_post_end = time.perf_counter()

        preprocess_ms = (t_pre_end - t_pre_start) * 1000.0
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        postprocess_ms = (t_post_end - t_post_start) * 1000.0
        total_ms = (t_post_end - t_pre_start) * 1000.0
        print(f"[Safety] Timing (ms): preprocess={preprocess_ms:.2f} inference={inference_ms:.2f} postprocess={postprocess_ms:.2f} total={total_ms:.2f}")

        results = []
        for b in range(len(bgr_list)):
            result_boxes: List[SafetyBox] = []
            for box, cl, score in zip(boxes_list[b], classes_list[b], scores_list[b]):
                x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
                result_boxes.append(SafetyBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=self.CLASSES[int(cl)],
                    conf=float(score),
                ))
            results.append(SafetyResult(boxes=result_boxes))

        return results
