import threading
import time
from typing import Optional

import cv2
import numpy as np
import slint

from camera_service import camera_service
from slintui.api_client import api_client


class DeskcleanViewport:
    """桌面清洁视图，负责检测工具是否就位以及桌面清洁程度"""

    def __init__(self):
        self.latest_result: FaceRecognizeResult = FaceRecognizeResult()
        self.latest_frame_bgr: np.ndarray | None = None

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
    @slint.callback(global_name=DeskcleanPageData)
    def request_deskclean_frame() -> None:
        # 拿原始帧
        frame = camera_service.get_frame()
        if frame is None:
            return

        # TODO: 叠加框

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_signin_viewport.latest_frame_bgr = frame
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.DeskcleanPageData.camera_frame = slint.Image.load_from_array(arr)

    @slint.callback
    async def deskclean_submit() -> None:
        frame_bgr = face_signin_viewport.latest_frame_bgr
        try:
            ok, buffer = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                raise RuntimeError("编码 JPEG 失败")

            result = {
                "sleeves_num": 0,
                "screwdriver_ready": window.DeskcleanPageData.screwdriver_ready,
                "wire_stripper_ready": window.DeskcleanPageData.wire_stripper_ready,
                "multimeter_ready": window.DeskcleanPageData.multimeter_ready,
                "crimping_ready": window.DeskcleanPageData.crimping_ready,
                "clean_progress": window.DeskcleanPageData.clean_progress,
            }

            response = await api_client.upload_deskclean_submit_async(buffer.tobytes(), result)

            if not response.get("success"):
                error = response.get("error", "未知")
                window.show_temporary_message(f"工位状态已提交: {error}")
                return

            window.show_temporary_message("工位状态已提交")
        except Exception as e:
            print(f"[Deskclean] 提交失败: {e}")
            window.show_temporary_message(f"工位状态已提交: {e}")

deskclean_viewport = DeskcleanViewport()
