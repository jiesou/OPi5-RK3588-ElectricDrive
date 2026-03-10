import threading
import time
from typing import Optional

import cv2
import numpy as np
import slint

from camera_service import camera_service


class DeskcleanViewport:
    """桌面清洁视图，负责检测工具是否就位以及桌面清洁程度"""

    def __init__(self):
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None

    def _inference_loop(self):
        print("[Deskclean] 推理线程启动")
        while self._running:
            time.sleep(0.1)
        print("[Deskclean] 推理线程退出")

    def start(self):
        if self._running:
            return
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    def stop(self):
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)


def bind_deskclean(window) -> None:
    @slint.callback
    def request_deskclean_frame() -> None:
        frame = camera_service.get_frame()
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.DeskcleanPageData.camera_frame = slint.Image.load_from_array(arr)

    window.DeskcleanPageData.request_deskclean_frame = request_deskclean_frame
    window.DeskcleanPageData.request_deskclean_frame()

    window.DeskcleanPageData.screwdriver_ready = False
    window.DeskcleanPageData.wire_stripper_ready = False
    window.DeskcleanPageData.multimeter_ready = False
    window.DeskcleanPageData.crimping_ready = False
    window.DeskcleanPageData.clean_progress = 0.0


deskclean_viewport = DeskcleanViewport()
