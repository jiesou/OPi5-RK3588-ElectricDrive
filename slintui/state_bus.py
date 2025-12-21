"""
状态总线 - 简化版，不依赖 Qt Signal
用于在 slintui 中管理帧数据和检测状态
"""
import numpy as np
from typing import Optional
import threading


class StateBus:
    """全局状态管理器"""

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def set_frame(self, frame: np.ndarray) -> None:
        """设置最新帧"""
        with self._lock:
            self._frame = frame.copy() if frame is not None else None

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧的副本"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# 全局单例
state_bus = StateBus()
