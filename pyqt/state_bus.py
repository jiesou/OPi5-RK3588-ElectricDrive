from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    terminal: int = 0
    cross: int = 0
    excopper: int = 0
    exterminal: int = 0


@dataclass
class Shot:
    detection: Detection = field(default_factory=Detection)


class StateBus(QObject):
    _frame: Optional[np.ndarray] = None
    detections_changed = Signal(Detection)
    shots_changed = Signal(Shot)
    # 视觉检测的详细结果（bounding boxes 等）
    detections_visual_changed = Signal(object)  # emit list[dict]
    # 视觉检测是否启用（从 UI 控制）
    inference_enabled_changed = Signal(bool)
    # 当前拍摄位置变化信号 (1, 2, 3)
    current_position_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self._current_detection: Detection = Detection()
        self._shots: List[Shot] = []  # list of Shot
        self._inference_enabled: bool = False
        self._current_position: int = 1  # 当前拍摄位置 1|2|3

    # frame
    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        return self._frame

    # detections
    def set_detections(self, detection: Detection) -> None:
        self._current_detection = detection
        self.detections_changed.emit(detection)
        
    def get_detections(self) -> Detection:
        return self._current_detection

    # shots
    def add_shot(self, shot: Shot) -> None:
        self._shots.append(shot)
        self.shots_changed.emit(shot)

    def clear_shots(self) -> None:
        self._shots = []
        self.shots_changed.emit(Shot())

    def get_shots(self) -> List[Shot]:
        return list(self._shots)

    def get_shots_totals(self) -> Shot:
        t = Shot()
        # accumulate into t.detection
        for s in self._shots:
            t.detection.terminal += int(s.detection.terminal)
            t.detection.cross += int(s.detection.cross)
        return t

    # inference enabled flag (UI toggles this)
    def set_inference_enabled(self, enabled: bool) -> None:
        self._inference_enabled = bool(enabled)
        self.inference_enabled_changed.emit(self._inference_enabled)

    def get_inference_enabled(self) -> bool:
        return bool(self._inference_enabled)

    # current position (1, 2, 3)
    def set_current_position(self, position: int) -> None:
        if position in (1, 2, 3):
            self._current_position = position
            self.current_position_changed.emit(self._current_position)

    def get_current_position(self) -> int:
        return self._current_position


stateBus = StateBus()
