from __future__ import annotations

from dataclasses import dataclass
from traceback import print_exception
from typing import List, Tuple

import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite as RKNN


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


class Yolo:
    """YOLO检测器，专为电拖装接评估场景设计 (RKNN版)"""
    def __init__(self):
        self.latest_result: YoloResult = YoloResult(
            detection=Detection(),
            boxes=[]
        )
        self.CLASSES = ("cross", "excopper", "exterminal", "terminal")
        self.IMG_SIZE = (640, 640)
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        
        model_path = "./batch3-rkfork-electricdrivev20.2.17.1.rknn"
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
            
    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        
        # Softmax
        y = np.exp(y)
        y = y / np.sum(y, axis=2, keepdims=True)
        
        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
        y = (y * acc_metrix).sum(2)
        return y

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.IMG_SIZE[1]//grid_h, self.IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        position = self.dfl(position)
        box_xy  = grid + 0.5 - position[:,0:2,:,:]
        box_xy2 = grid + 0.5 + position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        return np.array(keep)

    def _post_process_single(self, input_data):
        """对单个 batch 的 9 路输出做解码+NMS"""
        boxes, classes_conf, scores = [], [], []
        default_branch = 3
        pair_per_branch = len(input_data) // default_branch

        for i in range(default_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # NMS（单 batch 内）
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c_list = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c_list[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def post_process_batch(self, outputs, metas, orig_shape: Tuple[int, int]):
        """三张切图结果拼接后做全局 NMS"""
        if outputs is None or len(outputs) == 0:
            return None, None, None, None

        agg_boxes, agg_classes, agg_scores, agg_sources = [], [], [], []
        batch = min(len(metas), outputs[0].shape[0])
        orig_h, orig_w = orig_shape

        for i in range(batch):
            # 针对单个 batch，沿用原有后处理逻辑
            single_outputs = [o[i : i + 1] for o in outputs]
            boxes, classes, scores = self._post_process_single(single_outputs)
            if boxes is None:
                continue

            meta = metas[i]
            ratio = meta["ratio"]
            pad_x, pad_y = meta["pad"]
            offset_x, offset_y = meta["offset"]
            # 映射回原图坐标
            boxes = boxes.copy()
            boxes[:, 0] = (boxes[:, 0] - pad_x) / ratio + offset_x
            boxes[:, 1] = (boxes[:, 1] - pad_y) / ratio + offset_y
            boxes[:, 2] = (boxes[:, 2] - pad_x) / ratio + offset_x
            boxes[:, 3] = (boxes[:, 3] - pad_y) / ratio + offset_y

            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

            agg_boxes.append(boxes)
            agg_classes.append(classes)
            agg_scores.append(scores)
            agg_sources.append(np.full_like(classes, fill_value=meta["source"], dtype=np.int64))

        if not agg_boxes:
            return None, None, None, None

        boxes = np.concatenate(agg_boxes)
        classes = np.concatenate(agg_classes)
        scores = np.concatenate(agg_scores)
        sources = np.concatenate(agg_sources)

        # 全局 class-wise NMS
        f_boxes, f_classes, f_scores, f_sources = [], [], [], []
        for c in set(classes):
            inds = np.where(classes == c)[0]
            if len(inds) == 0:
                continue
            b = boxes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)
            if len(keep) == 0:
                continue
            kept_inds = inds[keep]
            f_boxes.append(boxes[kept_inds])
            f_classes.append(classes[kept_inds])
            f_scores.append(scores[kept_inds])
            f_sources.append(sources[kept_inds])

        if not f_boxes:
            return None, None, None, None

        return (
            np.concatenate(f_boxes),
            np.concatenate(f_classes),
            np.concatenate(f_scores),
            np.concatenate(f_sources),
        )

    def pre_process(self, bgr: np.ndarray):
        """三张切图 batch 预处理，返回 RGB batch 及映射参数"""
        h, w = bgr.shape[:2]

        # 左 720x720 -> 640x640
        left_crop = cv2.resize(bgr[0:720, 0:720], self.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        # 右 720x720 -> 640x640，重叠约 12.5%
        right_crop = cv2.resize(bgr[0:720, w - 720 : w], self.IMG_SIZE, interpolation=cv2.INTER_LINEAR)

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

        ratio_crop = self.IMG_SIZE[0] / 720.0  # 640 / 720
        metas = (
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (0.0, 0.0), "source": 0},
            {"ratio": ratio_crop, "pad": (0.0, 0.0), "offset": (w - 720.0, 0.0), "source": 1},
            {"ratio": 0.5, "pad": (0.0, 140.0), "offset": (0.0, 0.0), "source": 2},
        )

        return batch_input, metas

    def detect(self, bgr: np.ndarray) -> YoloResult:
        """
        执行检测并返回结果
        
        Args:
            bgr: BGR格式的图像数组
            
        Returns:
            YoloResult: 包含检测数量和边界框的结果
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

        print(f"Timing (ms): preprocess={preprocess_ms:.2f} inference={inference_ms:.2f} postprocess={postprocess_ms:.2f} total={total_ms:.2f}")
        
        if boxes is None:
            return self.latest_result

        final_boxes: List[Box] = []
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        
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
        
        detection = Detection(
            terminal=counts["terminal"],
            cross=counts["cross"],
            excopper=counts["excopper"],
            exterminal=counts["exterminal"]
        )

        if hasattr(self, 'last_push_result_time'):
            print(f"Time since last push: {time.perf_counter() - self.last_push_result_time:.2f} seconds")

        self.last_push_result_time = time.perf_counter()
        
        self.latest_result = YoloResult(detection=detection, boxes=final_boxes)
        return self.latest_result
    
# 全局单例
yolo = Yolo()
