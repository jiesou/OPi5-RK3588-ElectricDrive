from __future__ import annotations

from dataclasses import dataclass
from traceback import print_exception
from typing import List, Tuple

import numpy as np
import cv2
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
        
        model_path = "./electricdrivev20.2.17.1.rknn"
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

    def post_process(self, input_data):
        boxes, classes_conf, scores = [], [], []
        default_branch = 3
        pair_per_branch = len(input_data) // default_branch
        
        for i in range(default_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
            
        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)
        
        # NMS
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

    def letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

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

        # Preprocess
        img, ratio, (dw, dh) = self.letterbox(bgr, self.IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, 0) # Add batch dimension
        
        try:
            outputs = self.rknn.inference(inputs=[input_data])
            print(outputs)
            boxes, classes, scores = self.post_process(outputs)
        except Exception as e:
            print_exception(e)
            return self.latest_result
        
        if boxes is None:
            return self.latest_result

        final_boxes: List[Box] = []
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        
        for box, score, cl in zip(boxes, scores, classes):
            # Rescale boxes to original image
            x1, y1, x2, y2 = box
            x1 = (x1 - dw) / ratio[0]
            y1 = (y1 - dh) / ratio[1]
            x2 = (x2 - dw) / ratio[0]
            y2 = (y2 - dh) / ratio[1]
            
            # Clip to image bounds
            h, w = bgr.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            class_name = self.CLASSES[int(cl)]
            
            final_boxes.append(Box(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                label=class_name,
                conf=float(score)
            ))
            
            if class_name in counts:
                counts[class_name] += 1
        
        detection = Detection(
            terminal=counts["terminal"],
            cross=counts["cross"],
            excopper=counts["excopper"],
            exterminal=counts["exterminal"]
        )
        
        self.latest_result = YoloResult(detection=detection, boxes=final_boxes)
        return self.latest_result
    
# 全局单例
yolo = Yolo()
