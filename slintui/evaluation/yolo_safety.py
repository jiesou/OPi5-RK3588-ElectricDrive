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


@dataclass
class Track:
    """单个跟踪轨迹"""
    track_id: int
    box: SafetyBox
    age: int = 0
    hit_streak: int = 1
    miss_count: int = 0
    state: str = "tentative"

    def get_center(self) -> Tuple[float, float]:
        cx = (self.box.x1 + self.box.x2) / 2
        cy = (self.box.y1 + self.box.y2) / 2
        return cx, cy

    def get_area(self) -> float:
        return (self.box.x2 - self.box.x1) * (self.box.y2 - self.box.y1)


def _iou(box1: SafetyBox, box2: SafetyBox) -> float:
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    """简易 ByteTrack 实现，针对移动目标优化"""

    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_age = 10
        self.min_hits = 1
        self.iou_thresh = 0.35
        self.low_conf_thresh = 0.4

    def _match(self, boxes: List[SafetyBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks or not boxes:
            return [], list(range(len(self.tracks))), list(range(len(boxes)))

        iou_matrix = np.zeros((len(self.tracks), len(boxes)))
        for t_idx, track in enumerate(self.tracks):
            for b_idx, box in enumerate(boxes):
                if track.box.label == box.label:
                    iou_matrix[t_idx, b_idx] = _iou(track.box, box)

        matched = []
        matched_tracks = set()
        matched_boxes = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < self.iou_thresh:
                break

            t_idx, b_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((t_idx, b_idx))
            matched_tracks.add(t_idx)
            matched_boxes.add(b_idx)

            iou_matrix[t_idx, :] = 0
            iou_matrix[:, b_idx] = 0

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_boxes = [i for i in range(len(boxes)) if i not in matched_boxes]

        return matched, unmatched_tracks, unmatched_boxes

    def update(self, boxes: List[SafetyBox]) -> List[SafetyBox]:
        if not boxes:
            new_tracks = []
            for track in self.tracks:
                track.miss_count += 1
                track.age += 1
                if track.miss_count <= self.max_age:
                    new_tracks.append(track)
            self.tracks = new_tracks
            return []

        high_boxes = [b for b in boxes if b.conf >= self.low_conf_thresh]
        low_boxes = [b for b in boxes if b.conf < self.low_conf_thresh]

        matched1, unmatched_tracks, unmatched_high = self._match(high_boxes)

        for t_idx, b_idx in matched1:
            self.tracks[t_idx].box = high_boxes[b_idx]
            self.tracks[t_idx].hit_streak += 1
            self.tracks[t_idx].miss_count = 0
            self.tracks[t_idx].age += 1
            if self.tracks[t_idx].hit_streak >= self.min_hits:
                self.tracks[t_idx].state = "confirmed"

        if low_boxes and unmatched_tracks:
            unmatched_set = set(unmatched_tracks)
            temp_matched, temp_unmatched_tracks, _ = self._match(low_boxes)

            for t_idx, b_idx in temp_matched:
                if t_idx in unmatched_set:
                    self.tracks[t_idx].box = low_boxes[b_idx]
                    self.tracks[t_idx].miss_count = 0
                    self.tracks[t_idx].age += 1
                    unmatched_set.discard(t_idx)

            unmatched_tracks = list(unmatched_set)

        for t_idx in unmatched_tracks:
            self.tracks[t_idx].miss_count += 1
            self.tracks[t_idx].hit_streak = 0
            self.tracks[t_idx].age += 1

        for b_idx in unmatched_high:
            box = high_boxes[b_idx]
            self.tracks.append(Track(
                track_id=self.next_id,
                box=box,
                state="tentative"
            ))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_age]

        return [t.box for t in self.tracks if t.state == "confirmed" or t.hit_streak >= 1]


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
        self._trackers: List[ByteTracker] = []

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
            # 为每路摄像头维护独立 tracker
            while len(self._trackers) <= b:
                self._trackers.append(ByteTracker())

            result_boxes: List[SafetyBox] = []
            for box, cl, score in zip(boxes_list[b], classes_list[b], scores_list[b]):
                x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
                result_boxes.append(SafetyBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=self.CLASSES[int(cl)],
                    conf=float(score),
                ))
            # 跟踪平滑
            result_boxes = self._trackers[b].update(result_boxes)
            results.append(SafetyResult(boxes=result_boxes))

        return results
