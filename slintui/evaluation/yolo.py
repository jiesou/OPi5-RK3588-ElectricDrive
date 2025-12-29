from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from numba import njit, prange
from typing import List, Tuple, Deque
from collections import deque

import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite as RKNN

IMG_SIZE = (640, 640)

@dataclass
class Detection:
    """检测结果的数量统计"""
    terminal: int = 0
    cross: int = 0
    excopper: int = 0
    exterminal: int = 0


@dataclass
class Box:
    """单个检测框"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float
    source: int


@dataclass
class YoloResult:
    """完整的YOLO检测结果"""
    detection: Detection
    boxes: List[Box]

@njit(fastmath=True)
def _dfl(position):
    # Distribution Focal Loss (DFL)
    n, c = position.shape
    p_num = 4
    mc = c // p_num
    out = np.empty((n, 4), dtype=np.float32)

    for i in prange(n):
        for p in range(4):
            base = p * mc
            # Softmax
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
    # box_in: (n, 64, h, w)
    # cls_in: (n, nc, h, w)
    
    # Reshape & Transpose
    # (n, 64, h, w) -> (n, h*w, 64)
    box_raw = box_in.transpose(0, 2, 3, 1).reshape(n, -1, c)
    # (n, nc, h, w) -> (n, h*w, nc)
    cls_score = cls_in.transpose(0, 2, 3, 1).reshape(n, -1, cls_in.shape[1])
    
    # Grid
    col = np.arange(w)
    row = np.arange(h)
    col, row = np.meshgrid(col, row)
    grid = np.stack((col, row), axis=-1).reshape(1, -1, 2).astype(np.float32)
    
    stride = 640 / h
    
    return box_raw, cls_score, grid, stride


class Yolo:
    """YOLO检测器，专为电拖装接评估场景设计 (RKNN 版)"""
    def __init__(self):
        self.latest_result: YoloResult = YoloResult(
            detection=Detection(),
            boxes=[]
        )
        self.CLASSES = ("cross", "excopper", "exterminal", "terminal")
        self.OBJ_THRESH = 0.5
        self.NMS_THRESH = 0.5
        
        # 滤波相关
        self.FILTER_WINDOW = 5
        self.detection_history: Deque[Detection] = deque(maxlen=self.FILTER_WINDOW)
        
        model_path = "./batch3-rkfork-electricdrivev20.3.18.1.rknn"
        self.rknn = RKNN()
        print(f"Loading RKNN model: {model_path}")
        if self.rknn.load_rknn(model_path) != 0:
            print("Load RKNN model failed")
            self.rknn = None
            return
            
        if self.rknn.init_runtime() != 0:
            print("Init runtime environment failed")
            self.rknn = None
            return

    def post_process_batch(self, outputs, metas, orig_shape: Tuple[int, int]):
        """三张切图结果拼接后做全局 NMS"""
        if outputs is None or len(outputs) == 0:
            return None, None, None, None

        batch_size = outputs[0].shape[0]
        orig_h, orig_w = orig_shape
        
        # 提取所有尺度的原始数据 (延后 DFL)
        results = [_process_branch(outputs[i*3], outputs[i*3+1]) for i in range(3)]
        
        # 拼接所有尺度
        boxes_raw_cat = np.concatenate([r[0] for r in results], axis=1)
        scores_cat = np.concatenate([r[1] for r in results], axis=1)
        grids_cat = np.concatenate([np.tile(r[2], (batch_size, 1, 1)) for r in results], axis=1)
        strides_cat = np.concatenate([np.full((batch_size, r[0].shape[1], 1), r[3], dtype=np.float32) for r in results], axis=1)
        
        final_boxes = []
        final_scores = []
        final_classes = []
        final_sources = []

        for b in range(batch_size):
            meta = metas[b]
            ratio = meta["ratio"]
            pad_x, pad_y = meta["pad"]
            offset_x, offset_y = meta["offset"]
            source = meta["source"]
            
            b_scores = scores_cat[b]
            
            # 置信度过滤 (过阈值之后再做 DFL)
            class_ids = np.argmax(b_scores, axis=1)
            max_scores = np.max(b_scores, axis=1)
            
            mask = max_scores >= self.OBJ_THRESH
            if not np.any(mask):
                continue
                
            f_box_raw = boxes_raw_cat[b][mask]
            f_grid = grids_cat[b][mask]
            f_stride = strides_cat[b][mask]
            f_score = max_scores[mask]
            f_class = class_ids[mask]
            
            # DFL & Decode
            reg = _dfl(f_box_raw)
            
            lt = reg[:, 0:2]
            rb = reg[:, 2:4]
            
            f_grid_expanded = f_grid + 0.5
            x1y1 = (f_grid_expanded - lt) * f_stride
            x2y2 = (f_grid_expanded + rb) * f_stride
            
            f_box = np.concatenate((x1y1, x2y2), axis=1)
            
            # 映射坐标
            f_box[:, 0::2] = (f_box[:, 0::2] - pad_x) / ratio + offset_x
            f_box[:, 1::2] = (f_box[:, 1::2] - pad_y) / ratio + offset_y
            
            # 越界裁剪
            np.clip(f_box[:, 0], 0, orig_w, out=f_box[:, 0])
            np.clip(f_box[:, 1], 0, orig_h, out=f_box[:, 1])
            np.clip(f_box[:, 2], 0, orig_w, out=f_box[:, 2])
            np.clip(f_box[:, 3], 0, orig_h, out=f_box[:, 3])
            
            final_boxes.append(f_box)
            final_scores.append(f_score)
            final_classes.append(f_class)
            final_sources.append(np.full(len(f_box), source, dtype=np.int32))

        if not final_boxes:
            return None, None, None, None

        # 全部拼接
        all_boxes_cat = np.concatenate(final_boxes)
        all_scores_cat = np.concatenate(final_scores)
        all_classes_cat = np.concatenate(final_classes)
        all_sources_cat = np.concatenate(final_sources)

        # 基于 cv2.dnn.NMSBoxesBatched 的 NMS
        # xyxy to xywh
        boxes_wh = all_boxes_cat.copy()
        boxes_wh[:, 2] -= boxes_wh[:, 0] # w
        boxes_wh[:, 3] -= boxes_wh[:, 1] # h
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_wh, 
            all_scores_cat, 
            all_classes_cat, 
            self.OBJ_THRESH, 
            self.NMS_THRESH
        )
        
        if len(indices) == 0:
            return None, None, None, None
            
        indices = indices.flatten()
        
        return (
            all_boxes_cat[indices],
            all_classes_cat[indices],
            all_scores_cat[indices],
            all_sources_cat[indices],
        )

    def pre_process(self, bgr: np.ndarray):
        """三张切图 batch 预处理，返回 RGB batch 及映射参数"""
        h, w = bgr.shape[:2]

        # 左 720x720 -> 640x640
        left_crop = cv2.resize(bgr[0:720, 0:720], IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        # 右 720x720 -> 640x640，重叠约 12.5%
        right_crop = cv2.resize(bgr[0:720, w - 720 : w], IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # 整体 letterbox：按 0.5 缩放后上下各 140 padding
        scaled = cv2.resize(bgr, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_LINEAR)
        letter = cv2.copyMakeBorder(scaled, 140, 140, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 转 RGB 并打包 batch
        imgs = (
            cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(letter, cv2.COLOR_BGR2RGB),
        )
        batch_input = np.ascontiguousarray(np.stack(imgs, axis=0))

        ratio_crop = IMG_SIZE[0] / 720.0  # 640 / 720
        metas = (
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (0.0, 0.0), "source": 0},
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (w - 720.0, 0.0), "source": 1},
            {"ratio": 0.5, "pad": (0.0, 140.0), "offset": (0.0, 0.0), "source": 2},
        )

        return batch_input, metas

    def detect(self, bgr: np.ndarray) -> YoloResult:
        """
        执行检测并返回结果
        """
        if self.rknn is None:
            return self.latest_result

        h, w = bgr.shape[:2]

        # Timing: measure preprocess, inference, postprocess and total
        t_pre_start = time.perf_counter()
        input_data, metas = self.pre_process(bgr)
        t_pre_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()
        boxes, classes, scores, sources = self.post_process_batch(outputs, metas, (h, w))
        t_post_end = time.perf_counter()

        preprocess_ms = (t_pre_end - t_pre_start) * 1000.0
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        postprocess_ms = (t_post_end - t_post_start) * 1000.0
        total_ms = (t_post_end - t_pre_start) * 1000.0

        print(f"[Evaluation] Timing (ms): preprocess={preprocess_ms:.2f} inference={inference_ms:.2f} postprocess={postprocess_ms:.2f} total={total_ms:.2f}")
        
        final_boxes: List[Box] = []
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}

        if boxes is not None:
            for box, score, cl, src in zip(boxes, scores, classes, sources):
                x1, y1, x2, y2 = box

                class_name = self.CLASSES[int(cl)]

                final_boxes.append(Box(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    label=class_name,
                    conf=float(score),
                    source=int(src),
                ))
                
                if class_name in counts:
                    counts[class_name] += 1
        
        current_detection = Detection(
            terminal=counts["terminal"],
            cross=counts["cross"],
            excopper=counts["excopper"],
            exterminal=counts["exterminal"]
        )

        # 应用滤波 (滑动平均)
        self.detection_history.append(current_detection)
        
        avg_terminal = sum(d.terminal for d in self.detection_history) / len(self.detection_history)
        avg_cross = sum(d.cross for d in self.detection_history) / len(self.detection_history)
        avg_excopper = sum(d.excopper for d in self.detection_history) / len(self.detection_history)
        avg_exterminal = sum(d.exterminal for d in self.detection_history) / len(self.detection_history)
        
        filtered_detection = Detection(
            terminal=int(round(avg_terminal)),
            cross=int(round(avg_cross)),
            excopper=int(round(avg_excopper)),
            exterminal=int(round(avg_exterminal))
        )

        self.latest_result = YoloResult(detection=filtered_detection, boxes=final_boxes)
        return self.latest_result
yolo = Yolo()
