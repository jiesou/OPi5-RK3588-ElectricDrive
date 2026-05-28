from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from numba import njit, prange
from typing import List, Tuple, Deque, Optional
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
    total_ms: float = 0.0


@dataclass
class Track:
    """单个跟踪轨迹"""
    track_id: int
    box: Box
    age: int = 0  # 轨迹已存在帧数
    hit_streak: int = 1  # 连续命中次数
    miss_count: int = 0  # 连续丢失帧数
    state: str = "tentative"  # tentative, confirmed, lost

    def get_center(self) -> Tuple[float, float]:
        cx = (self.box.x1 + self.box.x2) / 2
        cy = (self.box.y1 + self.box.y2) / 2
        return cx, cy

    def get_area(self) -> float:
        return (self.box.x2 - self.box.x1) * (self.box.y2 - self.box.y1)


def _iou(box1: Box, box2: Box) -> float:
    """计算两个框的 IoU"""
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
    """简易 ByteTrack 实现"""

    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_age = 10  # 最大丢失帧数后删除
        self.min_hits = 2  # 确认轨迹需要的连续命中次数
        self.iou_thresh = 0.5  # IoU 匹配阈值
        self.low_conf_thresh = 0.6  # 低置信度阈值

    def _match(self, boxes: List[Box]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """基于 IoU 的贪心匹配
        返回: (匹配对列表, 未匹配轨迹索引, 未匹配检测框索引)
        """
        if not self.tracks or not boxes:
            return [], list(range(len(self.tracks))), list(range(len(boxes)))

        # 计算 IoU 矩阵
        iou_matrix = np.zeros((len(self.tracks), len(boxes)))
        for t_idx, track in enumerate(self.tracks):
            for b_idx, box in enumerate(boxes):
                if track.box.label == box.label:  # 同类别才匹配
                    iou_matrix[t_idx, b_idx] = _iou(track.box, box)

        # 贪心匹配
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

            # 清除已匹配的行列
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, b_idx] = 0

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_boxes = [i for i in range(len(boxes)) if i not in matched_boxes]

        return matched, unmatched_tracks, unmatched_boxes

    def update(self, boxes: List[Box]) -> List[Box]:
        """更新跟踪器并返回平滑后的检测框"""
        if not boxes:
            # 无检测结果，所有轨迹丢失计数+1
            new_tracks = []
            for track in self.tracks:
                track.miss_count += 1
                track.age += 1
                if track.miss_count <= self.max_age:
                    new_tracks.append(track)
            self.tracks = new_tracks
            return []

        # 分离高/低置信度检测框
        high_boxes = [b for b in boxes if b.conf >= self.low_conf_thresh]
        low_boxes = [b for b in boxes if b.conf < self.low_conf_thresh]

        # 第一轮：高置信度匹配
        matched1, unmatched_tracks, unmatched_high = self._match(high_boxes)

        # 更新匹配的轨迹
        for t_idx, b_idx in matched1:
            self.tracks[t_idx].box = high_boxes[b_idx]
            self.tracks[t_idx].hit_streak += 1
            self.tracks[t_idx].miss_count = 0
            self.tracks[t_idx].age += 1
            if self.tracks[t_idx].hit_streak >= self.min_hits:
                self.tracks[t_idx].state = "confirmed"

        # 第二轮：低置信度与未匹配轨迹匹配
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

        # 处理未匹配轨迹
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].miss_count += 1
            self.tracks[t_idx].hit_streak = 0
            self.tracks[t_idx].age += 1

        # 创建新轨迹
        for b_idx in unmatched_high:
            box = high_boxes[b_idx]
            self.tracks.append(Track(
                track_id=self.next_id,
                box=box,
                state="tentative"
            ))
            self.next_id += 1

        # 删除过期轨迹
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_age]

        # 返回确认状态的轨迹
        return [t.box for t in self.tracks if t.state == "confirmed" or t.hit_streak >= 1]

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
    """YOLO检测器，专为电气装接评估场景设计 (RKNN 版)"""
    def __init__(self):
        self.tile_inference_enabled = False
        self.latest_result: YoloResult = YoloResult(
            detection=Detection(),
            boxes=[]
        )
        self.latest_crop_bgr: np.ndarray | None = None
        self.CLASSES = ( "terminal", ) # ("cross", "excopper", "exterminal", "terminal")
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.7
        
        # 滤波相关
        self.FILTER_WINDOW = 5
        self.detection_history: Deque[Detection] = deque(maxlen=self.FILTER_WINDOW)

        # ByteTrack 跟踪器
        self.tracker = ByteTracker()
        
        model_path = "./batch3-rkfork-electricv322valopt.rknn"
        self.rknn = RKNN()
        print(f"Loading RKNN model: {model_path}")
        if self.rknn.load_rknn(model_path) != 0:
            print("Load RKNN model failed")
            self.rknn = None
            return
            
        if self.rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2) != 0:
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

    EXPECTED_H = 1080
    EXPECTED_W = 1920
    CROP_SIZE = 640
    DISP_W_RATIO = 0.51
    DISP_H_RATIO = 0.61

    def pre_process(self, bgr: np.ndarray):
        """左下角裁切，上下半分 + 全局，三图并行推理"""
        orig_h, orig_w = bgr.shape[:2]
        scale_fit, pad_letter_x, pad_letter_y = 1.0, 0.0, 0.0

        if orig_h != self.EXPECTED_H or orig_w != self.EXPECTED_W:
            scale_h = self.EXPECTED_H / orig_h
            scale_w = self.EXPECTED_W / orig_w
            scale_fit = min(scale_h, scale_w)

            new_w = int(orig_w * scale_fit)
            new_h = int(orig_h * scale_fit)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = self.EXPECTED_W - new_w
            pad_h = self.EXPECTED_H - new_h
            pad_left = pad_w // 2
            pad_top = pad_h // 2

            if pad_w > 0 or pad_h > 0:
                bgr = cv2.copyMakeBorder(bgr, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))

            pad_letter_x = float(pad_left)
            pad_letter_y = float(pad_top)
            print(f"[YOLO] 输入 {orig_w}x{orig_h} 已 letterbox 到 {self.EXPECTED_W}x{self.EXPECTED_H}")

        h, w = bgr.shape[:2]
        C = self.CROP_SIZE
        half = C // 2
        pad_size = (C - half) // 2  # 160

        crop = bgr[h - C:h, 0:C]

        top_padded = cv2.copyMakeBorder(crop[0:half, 0:C], pad_size, pad_size, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bottom_padded = cv2.copyMakeBorder(crop[half:C, 0:C], pad_size, pad_size, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))

        imgs = (
            cv2.cvtColor(top_padded, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bottom_padded, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        )
        batch_input = np.ascontiguousarray(np.stack(imgs, axis=0))

        metas = (
            {"ratio": scale_fit, "pad": (pad_letter_x, pad_letter_y + pad_size),
             "offset": (0.0, (h - C) / scale_fit), "source": 0},
            {"ratio": scale_fit, "pad": (pad_letter_x, pad_letter_y + pad_size),
             "offset": (0.0, (h - half) / scale_fit), "source": 1},
            {"ratio": scale_fit, "pad": (pad_letter_x, pad_letter_y),
             "offset": (0.0, (h - C) / scale_fit), "source": 2},
        )

        return batch_input, metas, (orig_h, orig_w), crop

    def detect(self, bgr: np.ndarray) -> YoloResult:
        """
        执行检测并返回结果
        """
        if self.rknn is None:
            return self.latest_result

        t_pre_start = time.perf_counter()
        input_data, metas, orig_shape, _crop_inference = self.pre_process(bgr)
        t_pre_end = time.perf_counter()

        orig_h, orig_w = orig_shape
        disp_w = int(orig_w * self.DISP_W_RATIO)
        disp_h = int(orig_h * self.DISP_H_RATIO)
        disp_y0 = orig_h - disp_h
        disp_crop = bgr[disp_y0:orig_h, 0:disp_w].copy()

        t_inf_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()
        boxes, classes, scores, sources = self.post_process_batch(outputs, metas, orig_shape)
        t_post_end = time.perf_counter()

        # 构建检测框 + ByteTrack 跟踪
        t_track_start = time.perf_counter()

        # 将框框坐标从原始帧空间映射到显示裁切空间（左下角 DISP_W_RATIO × DISP_H_RATIO）
        raw_boxes: List[Box] = []
        if boxes is not None:
            for box, score, cl, src in zip(boxes, scores, classes, sources):
                x1, y1, x2, y2 = box
                x1 = max(0, min(disp_w, int(x1)))
                y1 = max(0, min(disp_h, int(y1 - disp_y0)))
                x2 = max(0, min(disp_w, int(x2)))
                y2 = max(0, min(disp_h, int(y2 - disp_y0)))
                class_name = self.CLASSES[int(cl)]
                raw_boxes.append(Box(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=class_name,
                    conf=float(score),
                    source=int(src),
                ))
        final_boxes = self.tracker.update(raw_boxes)

        t_track_end = time.perf_counter()

        if not self.tile_inference_enabled:
            final_boxes = [b for i, b in enumerate(final_boxes) if i % 3 == 0]
            time.sleep(0.2)

        preprocess_ms = (t_pre_end - t_pre_start) * 1000.0
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        postprocess_ms = (t_post_end - t_post_start) * 1000.0
        track_ms = (t_track_end - t_track_start) * 1000.0
        total_ms = (t_track_end - t_pre_start) * 1000.0

        print(f"[Evaluation] Timing (ms): preprocess={preprocess_ms:.2f} inference={inference_ms:.2f} postprocess={postprocess_ms:.2f} track={track_ms:.2f} total={total_ms:.2f}")

        # 统计
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        for box in final_boxes:
            if box.label in counts:
                counts[box.label] += 1
        
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

        self.latest_result = YoloResult(detection=filtered_detection, boxes=final_boxes, total_ms=total_ms)
        self.latest_crop_bgr = disp_crop
        return self.latest_result
yolo = Yolo()
