from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float


@dataclass(frozen=True)
class DetectionResult:
    boxes: List[Box]
    counts: Dict[str, int]

class Yolo:
    """Very small YOLO wrapper.

    - If an Ultralytics-compatible model exists, it will be loaded lazily.
    - If not, detection returns empty results.
    """

    def __init__(self):
        self._load_error: Optional[str] = None

    def _ensure_model(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            self._load_error = f"ultralytics not available: {e}"
            self.model_path = None
            return None

        try:
            self._model = YOLO("electricdrivev10.3.15.2_rknn_model", task='detect')
            self._load_error = None
            return self._model
        except Exception as e:
            # If model loading fails (unsupported format/runtime), disable inference.
            self._load_error = f"failed to load model '{self.model_path}': {e}"
            self.model_path = None
            return None

    def last_error(self) -> Optional[str]:
        return self._load_error

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        """Run detection and return boxes + counts.

        The returned counts use these keys:
        - terminal
        - cross
        - excopper
        - exterminal
        """

        counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}

        model = self._ensure_model()
        if model is None:
            return DetectionResult(boxes=[], counts=counts)

        # Ultralytics expects RGB arrays.
        rgb = bgr[:, :, ::-1]
        results = model.predict(rgb,
                conf=0.05,   # 保留置信度 ≥ 0.05 的检测框
                iou=0.2,    # NMS 的 IoU 阈值设为 0.3，不那么容易重叠
                verbose=True
            )
        if not results:
            return DetectionResult(boxes=[], counts=counts)

        r0 = results[0]

        boxes: List[Box] = []
        names = getattr(r0, "names", None) or getattr(model, "names", {})

        # r0.boxes may be None
        if getattr(r0, "boxes", None) is None:
            return DetectionResult(boxes=[], counts=counts)

        for b in r0.boxes:
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            label = str(names.get(cls_id, cls_id))
            boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, label=label, conf=conf))

            # Map label to our business keys (keep it explicit and simple).
            key = _label_to_key(label)
            if key in counts:
                counts[key] += 1

        return DetectionResult(boxes=boxes, counts=counts)


def _label_to_key(label: str) -> str:
    s = label.lower().strip()
    # You can refine mapping based on your trained class names.
    if "cross" in s or "jiaoch" in s:
        return "cross"
    if "excopper" in s or "copper" in s or "lout" in s:
        return "excopper"
    if "exterminal" in s or "terminal_ex" in s or "loud" in s:
        return "exterminal"
    if "terminal" in s or "sleeve" in s or "haomaguan" in s:
        return "terminal"
    return s


# module singleton (simple to import/use)
yolo = Yolo()
