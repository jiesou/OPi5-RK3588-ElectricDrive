from dataclasses import dataclass
import threading
import time

import cv2
from cv2.typing import MatLike
import numpy as np
import slint

from camera_service import camera_service
from api_client import api_client


@dataclass
class DeskcleanDetectResult:
    """桌面清洁检测结果"""
    clutter_ratio: float = 0.0
    clutter_mask: MatLike | None = None
    desk_region: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)


class DeskcleanViewport:
    """工位清洁视图，负责检测工具是否就位以及桌面清洁程度"""

    def __init__(self):
        self.latest_result: DeskcleanDetectResult = DeskcleanDetectResult()
        self.latest_frame_bgr: np.ndarray | None = None

        self._running = False
        self._inference_thread: threading.Thread | None = None

    def _detect_desk_clutter(self, frame: np.ndarray) -> DeskcleanDetectResult:
        """检测桌面杂物并计算占比

        思路：
        1. 裁剪桌面区域
        2. 转灰度 + 高斯模糊
        3. 边缘检测 (Canny)
        4. 形态学操作连接边缘
        5. 计算边缘区域占比作为"杂物占比"
        """
        if frame is None:
            return DeskcleanDetectResult()

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (0.3, 0.6, 0.7, 0.8)  # 桌面区域 (相对坐标)
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        # 裁剪桌面区域
        desk_img = frame[y1:y2, x1:x2]
        if desk_img.size == 0:
            return DeskcleanDetectResult()

        # 转灰度并模糊
        gray = cv2.cvtColor(desk_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 形态学操作：闭运算连接相邻边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 再做一次膨胀扩大区域
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # 计算杂物占比
        clutter_pixels = np.count_nonzero(dilated)
        total_pixels = dilated.size
        clutter_ratio = clutter_pixels / total_pixels

        # 对数归一化：log(1 + x) 让小值也有分数，减少 0 分情况
        adjusted = max(0.0, clutter_ratio - 0.02)
        normalized_ratio = min(1.0, np.log1p(adjusted * 30) / np.log1p(0.3 * 15))

        # 创建彩色掩码用于可视化
        clutter_mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask_region = clutter_mask[y1:y2, x1:x2]
        mask_region[dilated > 0] = [0, 0, 255]  # 红色标记杂物区域

        return DeskcleanDetectResult(
            clutter_ratio=normalized_ratio,
            clutter_mask=clutter_mask,
            desk_region=(x1, y1, x2, y2),
        )

    def _draw_overlay(self, frame: np.ndarray, result: DeskcleanDetectResult) -> np.ndarray:
        """在画面上绘制识别结果可视化"""
        if frame is None or result is None:
            return frame

        overlay = frame.copy()

        # 绘制杂物区域掩码
        if result.clutter_mask is not None:
            cv2.addWeighted(overlay, 0.7, result.clutter_mask, 0.5, 0, overlay)

        # 绘制桌面区域框
        if result.desk_region:
            x1, y1, x2, y2 = result.desk_region
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return overlay

    def _inference_loop(self):
        print("[Deskclean] 推理线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is not None:
                self.latest_frame_bgr = frame.copy()
                t_start = time.perf_counter()
                self.latest_result = self._detect_desk_clutter(frame)
                t_end = time.perf_counter()
                print(f"[Deskclean] Timing (ms): detect={(t_end - t_start)*1000:.1f}")

            time.sleep(0.1)  # 每 100ms 检测一次
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
    # 启动推理线程
    deskclean_viewport.start()

    @slint.callback(global_name="DeskcleanPageData")
    def request_deskclean_frame() -> None:
        # 拿原始帧
        frame = camera_service.get_frame()
        if frame is None:
            return

        # 获取检测结果并叠加可视化
        result = deskclean_viewport.latest_result
        overlay_frame = deskclean_viewport._draw_overlay(frame, result)

        # 更新 UI 的清洁进度
        clean_progress = 1.0 - result.clutter_ratio

        window.DeskcleanPageData.clean_progress = clean_progress

        rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        deskclean_viewport.latest_frame_bgr = frame  # 保存原始帧用于提交
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

    # 手动绑定回调到 window
    window.DeskcleanPageData.request_deskclean_frame = request_deskclean_frame
    window.DeskcleanPageData.deskclean_submit = deskclean_submit

deskclean_viewport = DeskcleanViewport()
