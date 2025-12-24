from __future__ import annotations

from dataclasses import dataclass
from traceback import print_exception
from typing import List

import numpy as np


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
    """YOLO检测器，专为电拖装接评估场景设计"""
    
    def __init__(self):
        self.latest_result: YoloResult = YoloResult(
            detection=Detection(),
            boxes=[]
        )
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO("electricdrivev10.3.15.2_rknn_model", task='detect')
        except Exception as e:
            print_exception(e)
    
    def detect(self, bgr: np.ndarray) -> YoloResult:
        """
        执行检测并返回结果
        
        Args:
            bgr: BGR格式的图像数组
            
        Returns:
            YoloResult: 包含检测数量和边界框的结果
        """
        if not hasattr(self, "model"):
            return self.latest_result

        # Ultralytics 需要 RGB 格式
        rgb = bgr[:, :, ::-1]
        
        try:
            yolo_results = self.model.predict(
                rgb,
                conf=0.2,   # 保留置信度 ≥ 0.05 的检测框
                iou=0.45,     # NMS 的 IoU 阈值
                verbose=True
            )
        except Exception as e:
            print_exception(e)
            return self.latest_result
        
        if not yolo_results:
            return self.latest_result

        yolo_result0 = yolo_results[0]
        
        # yolo_result0.boxes 可能为 None
        if getattr(yolo_result0, "boxes", None) is None:
            return self.latest_result

        boxes: List[Box] = []
        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        
        for box in yolo_result0.boxes:
            class_name = yolo_result0.names[int(box.cls[0])]
            
            # 类名映射
            if class_name == "class3":
                class_name = "terminal"
            elif class_name == "class2":
                class_name = "exterminal"
            elif class_name == "class1":
                class_name = "excopper"
            elif class_name == "class0":
                class_name = "cross"
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            boxes.append(Box(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                label=class_name,
                conf=conf
            ))
            
            # 统计数量
            if class_name in counts:
                counts[class_name] += 1
        
        detection = Detection(
            terminal=counts["terminal"],
            cross=counts["cross"],
            excopper=counts["excopper"],
            exterminal=counts["exterminal"]
        )
        
        self.latest_result = YoloResult(detection=detection, boxes=boxes)
        return self.latest_result
    
# 全局单例
yolo = Yolo()
