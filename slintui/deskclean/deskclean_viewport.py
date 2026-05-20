from dataclasses import dataclass
import threading
import time

import cv2
from cv2.typing import MatLike
import numpy as np
import slint

from camera_service import camera_service
from api_client import api_client
from .yolo_tools import yolo_tools, ToolBox

TOOL_COLORS = {
    "multimeter": (255, 0, 0),     # 蓝色
    "screwdriver": (0, 255, 0),    # 绿色
    "wirestripper": (0, 165, 255), # 橙色
    "crimping": (0, 0, 255),       # 红色
}


@dataclass
class DeskcleanDetectResult:
    """桌面清洁检测结果"""
    clutter_ratio: float = 0.0
    clutter_mask: MatLike | None = None
    desk_region: tuple[int, int, int, int] | None = None


class DeskcleanViewport:
    """工位清洁视图，负责检测工具是否就位以及桌面清洁程度"""

    def __init__(self):
        self.latest_result = DeskcleanDetectResult()
        self._result_lock = threading.Lock()
        self.latest_frame_bgr: np.ndarray | None = None

        self._running = False
        self._yolo_thread: threading.Thread | None = None
        self._clutter_thread: threading.Thread | None = None

    @staticmethod
    def _detect_desk_clutter(frame: np.ndarray) -> DeskcleanDetectResult:

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (0.6, 0.5, 0.95, 0.95)  # 桌面区域 (相对坐标)
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        desk_img = frame[y1:y2, x1:x2]
        if desk_img.size == 0:
            return DeskcleanDetectResult()

        gray = cv2.cvtColor(desk_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        clutter_pixels = np.count_nonzero(dilated)
        total_pixels = dilated.size
        clutter_ratio = clutter_pixels / total_pixels

        adjusted = max(0.0, clutter_ratio - 0.02)
        normalized_ratio = min(1.0, np.log1p(adjusted * 30) / np.log1p(0.3 * 15))

        clutter_mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask_region = clutter_mask[y1:y2, x1:x2]
        mask_region[dilated > 0] = [0, 0, 255]

        return DeskcleanDetectResult(
            clutter_ratio=normalized_ratio,
            clutter_mask=clutter_mask,
            desk_region=(x1, y1, x2, y2),
        )

    @staticmethod
    def _draw_tool_boxes(overlay: np.ndarray, tool_boxes: list[ToolBox]) -> None:
        for box in tool_boxes:
            color = TOOL_COLORS.get(box.label, (255, 255, 255))
            cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            label = f"{box.label} {box.conf:.2f}"
            cv2.putText(overlay, label, (box.x1, max(20, box.y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @staticmethod
    def _draw_overlay(frame: np.ndarray, result: DeskcleanDetectResult,
                      tool_boxes: list[ToolBox] | None) -> np.ndarray:
        if frame is None:
            return frame

        overlay = frame.copy()

        if result.clutter_mask is not None:
            cv2.addWeighted(overlay, 0.7, result.clutter_mask, 0.5, 0, overlay)

        # 绿色边框
        # if result.desk_region:
        #     x1, y1, x2, y2 = result.desk_region
        #     cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if tool_boxes:
            DeskcleanViewport._draw_tool_boxes(overlay, tool_boxes)

        return overlay

    def _yolo_loop(self):
        print("[Deskclean] YOLO 推理线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is not None:
                t0 = time.perf_counter()
                yolo_tools.detect(frame)
                t1 = time.perf_counter()
                print(f"[Deskclean] YOLO Timing (ms): {(t1 - t0)*1000:.1f}")
            time.sleep(0.1)
        print("[Deskclean] YOLO 推理线程退出")

    def _desk_clutter_loop(self):
        print("[Deskclean] 杂物检测线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is not None:
                t0 = time.perf_counter()
                result = self._detect_desk_clutter(frame)
                t1 = time.perf_counter()
                with self._result_lock:
                    self.latest_result = result
                print(f"[Deskclean] Clutter Timing (ms): {(t1 - t0)*1000:.1f}")
            time.sleep(0.1)
        print("[Deskclean] 杂物检测线程退出")

    def start(self):
        if self._running:
            return
        self._running = True
        self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
        self._clutter_thread = threading.Thread(target=self._desk_clutter_loop, daemon=True)
        self._yolo_thread.start()
        self._clutter_thread.start()

    def stop(self):
        self._running = False
        for t in (self._yolo_thread, self._clutter_thread):
            if t:
                t.join(timeout=1.0)

    def get_latest_result(self) -> DeskcleanDetectResult:
        with self._result_lock:
            return self.latest_result


def bind_deskclean(window) -> None:
    deskclean_viewport.start()

    @slint.callback(global_name="DeskcleanPageData")
    def request_deskclean_frame() -> None:
        frame = camera_service.get_frame()
        if frame is None:
            return

        result = deskclean_viewport.get_latest_result()
        tool_result = yolo_tools.latest_result
        overlay_frame = DeskcleanViewport._draw_overlay(frame, result, tool_result.boxes)

        clean_progress = 1.0 - result.clutter_ratio
        window.DeskcleanPageData.clean_progress = clean_progress

        tool_result = yolo_tools.latest_result
        present = set(tool_result.present)
        window.DeskcleanPageData.screwdriver_ready = "screwdriver" in present
        window.DeskcleanPageData.wire_stripper_ready = "wirestripper" in present
        window.DeskcleanPageData.multimeter_ready = "multimeter" in present
        window.DeskcleanPageData.crimping_ready = "crimping" in present

        deskclean_viewport.latest_frame_bgr = overlay_frame
        rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.DeskcleanPageData.camera_frame = slint.Image.load_from_array(arr)

    @slint.callback(global_name="DeskcleanPageData")
    async def deskclean_submit() -> None:
        frame_bgr = deskclean_viewport.latest_frame_bgr
        try:
            print("[DeskClean] 正在提交工位状态")
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
                print(f"[DeskClean] 提交失败: {error}")
                window.show_temporary_message(f"工位状态已提交: {error}")
                return

            window.show_temporary_message("工位状态已提交")
            print(f"[DeskClean] 工位状态已提交。服务器响应: {response}")
        except Exception as e:
            print(f"[DeskClean] 提交失败: {e}")
            window.show_temporary_message(f"工位状态已提交: {e}")

    window.DeskcleanPageData.request_deskclean_frame = request_deskclean_frame
    window.DeskcleanPageData.deskclean_submit = deskclean_submit


deskclean_viewport = DeskcleanViewport()
