"""YOLO v11 Nano 工具检测器，基于 RKNN NPU 推理

输入: 摄像头帧（BGR）
预处理: 裁切中上区域 → 640x320 → letterbox 640x640
后处理: DFL 解码 → NMS → 坐标映射回原图
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2
from rknnlite.api import RKNNLite as RKNN

IMG_SIZE = (640, 640)
CLASSES = ("multimeter", "screwdriver", "wirestripper", "crimping")
OBJ_THRESH = 0.5
NMS_THRESH = 0.5


@dataclass
class ToolBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    label_id: int
    conf: float


@dataclass
class ToolDetection:
    boxes: List[ToolBox]
    present: List[str]


class SimpleTracker:
    def __init__(self):
        self.tracks: List[dict] = []
        self.next_id = 0
        self.iou_thresh = 0.3
        self.max_lost = 10

    @staticmethod
    def _iou(a: ToolBox, b: ToolBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes: List[ToolBox]) -> List[ToolBox]:
        if not boxes:
            for t in self.tracks:
                t["lost"] += 1
            self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]
            return []

        matched_t = set()
        matched_b = set()

        for t_idx, track in enumerate(self.tracks):
            best_iou = 0.0
            best_b = -1
            for b_idx, box in enumerate(boxes):
                if b_idx in matched_b or track["box"].label != box.label:
                    continue
                i = self._iou(track["box"], box)
                if i > best_iou:
                    best_iou = i
                    best_b = b_idx
            if best_iou > self.iou_thresh:
                matched_t.add(t_idx)
                matched_b.add(best_b)
                self.tracks[t_idx]["box"] = boxes[best_b]
                self.tracks[t_idx]["lost"] = 0
                self.tracks[t_idx]["age"] += 1

        for t_idx in range(len(self.tracks)):
            if t_idx not in matched_t:
                self.tracks[t_idx]["lost"] += 1
                self.tracks[t_idx]["age"] += 1

        for b_idx, box in enumerate(boxes):
            if b_idx not in matched_b:
                self.tracks.append({
                    "id": self.next_id,
                    "box": box,
                    "age": 0,
                    "lost": 0,
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]
        return [t["box"] for t in self.tracks if t["lost"] == 0 and t["age"] >= 0]


def _dfl_decode(position: np.ndarray) -> np.ndarray:
    n, c = position.shape
    p_num = 4
    mc = c // p_num
    out = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        for p in range(p_num):
            vals = position[i, p * mc : (p + 1) * mc].astype(np.float64)
            vals -= vals.max()
            exp_vals = np.exp(vals)
            s = exp_vals.sum()
            out[i, p] = float(np.dot(exp_vals, np.arange(mc, dtype=np.float64)) / s)
    return out


class YoloTools:
    def __init__(self, model_path: str = "./batch1-electricdrive-tools-v10.rknn"):
        self.tracker = SimpleTracker()
        self.latest_result: ToolDetection = ToolDetection(boxes=[], present=[])

        self.rknn = RKNN()
        print(f"[YoloTools] Loading RKNN model: {model_path}")
        if self.rknn.load_rknn(model_path) != 0:
            print("[YoloTools] Load RKNN model failed")
            self.rknn = None
            return
        if self.rknn.init_runtime() != 0:
            print("[YoloTools] Init runtime environment failed")
            self.rknn = None
            return
        print("[YoloTools] Model loaded successfully")

    def pre_process(self, bgr: np.ndarray):
        orig_h, orig_w = bgr.shape[:2]

        crop_w = orig_w // 2
        crop_h = min(orig_w // 4, orig_h)

        x1 = (orig_w - crop_w) // 2
        y1 = 0

        crop = bgr[y1 : y1 + crop_h, x1 : x1 + crop_w]
        resized = cv2.resize(crop, (640, 320), interpolation=cv2.INTER_LINEAR)

        pad_y = 160
        letterbox = cv2.copyMakeBorder(
            resized, pad_y, pad_y, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)
        input_data = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.float32)

        meta = {
            "x1": x1,
            "y1": y1,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "pad_y": pad_y,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }
        return input_data, meta

    def post_process(self, outputs, meta) -> ToolDetection:
        if outputs is None or len(outputs) == 0:
            return ToolDetection(boxes=[], present=[])

        x1 = meta["x1"]
        y1 = meta["y1"]
        crop_w = meta["crop_w"]
        crop_h = meta["crop_h"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]

        all_boxes = []
        all_scores = []
        all_classes = []

        for i in range(3):
            box_out = outputs[i * 2]      # (1, 64, h, w)
            cls_out = outputs[i * 2 + 1]  # (1, nc, h, w)

            _, _, h, w = box_out.shape
            nc = cls_out.shape[1]

            box_raw = box_out.transpose(0, 2, 3, 1).reshape(1, -1, 64)
            cls_raw = cls_out.transpose(0, 2, 3, 1).reshape(1, -1, nc)

            col, row = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((col, row), axis=-1).reshape(1, -1, 2).astype(np.float32)
            stride = 640.0 / h

            scores = cls_raw[0]
            boxes_ = box_raw[0]
            grid_ = grid[0]

            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)

            mask = max_scores >= OBJ_THRESH
            if not np.any(mask):
                continue

            f_box = boxes_[mask]
            f_grid = grid_[mask]
            f_score = max_scores[mask]
            f_class = class_ids[mask]

            reg = _dfl_decode(f_box)

            g = f_grid + 0.5
            x1y1 = (g - reg[:, 0:2]) * stride
            x2y2 = (g + reg[:, 2:4]) * stride
            decoded = np.concatenate((x1y1, x2y2), axis=1)

            all_boxes.append(decoded)
            all_scores.append(f_score)
            all_classes.append(f_class)

        if not all_boxes:
            return ToolDetection(boxes=[], present=[])

        boxes_cat = np.concatenate(all_boxes)
        scores_cat = np.concatenate(all_scores)
        classes_cat = np.concatenate(all_classes)

        boxes_wh = boxes_cat.copy()
        boxes_wh[:, 2] -= boxes_wh[:, 0]
        boxes_wh[:, 3] -= boxes_wh[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(), scores_cat.tolist(), OBJ_THRESH, NMS_THRESH
        )
        if indices is None or len(indices) == 0:
            return ToolDetection(boxes=[], present=[])

        indices = np.array(indices).flatten()

        nms_boxes = boxes_cat[indices]
        nms_scores = scores_cat[indices]
        nms_classes = classes_cat[indices]

        tool_boxes: List[ToolBox] = []
        present: List[str] = []

        scale_x = crop_w / 640.0
        scale_y = crop_h / 320.0

        for box, score, cls_id in zip(nms_boxes, nms_scores, nms_classes):
            bx = float(box[0]) * scale_x + x1
            by = (float(box[1]) - pad_y) * scale_y + y1
            bw = (float(box[2]) - float(box[0])) * scale_x
            bh = (float(box[3]) - float(box[1])) * scale_y

            bx = max(0.0, min(float(orig_w), bx))
            by = max(0.0, min(float(orig_h), by))
            bx2 = max(0.0, min(float(orig_w), bx + bw))
            by2 = max(0.0, min(float(orig_h), by + bh))

            label = CLASSES[int(cls_id)]
            tool_boxes.append(ToolBox(
                x1=int(bx), y1=int(by), x2=int(bx2), y2=int(by2),
                label=label, label_id=int(cls_id), conf=float(score),
            ))

        tracked_boxes = self.tracker.update(tool_boxes)
        present = list({b.label for b in tracked_boxes})

        return ToolDetection(boxes=tracked_boxes, present=present)

    def detect(self, bgr: np.ndarray) -> ToolDetection:
        if self.rknn is None:
            return ToolDetection(boxes=[], present=[])

        input_data, meta = self.pre_process(bgr)
        outputs = self.rknn.inference(inputs=[input_data])
        result = self.post_process(outputs, meta)
        self.latest_result = result
        return result


yolo_tools = YoloTools()
