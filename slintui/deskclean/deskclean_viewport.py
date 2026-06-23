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
    clutter_count: int = 0
    clutter_mask: MatLike | None = None
    desk_region: tuple[int, int, int, int] | None = None


@dataclass
class TrackedClutter:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class ClutterTracker:
    """IoU 匹配 + 连续帧确认的杂物消抖跟踪器"""

    def __init__(self, min_hits: int = 3, max_misses: int = 5, min_iou: float = 0.3):
        self.tracks: list[TrackedClutter] = []
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.min_iou = min_iou

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def update(self, contours: list[np.ndarray], offset_x: int, offset_y: int):
        det_bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            det_bboxes.append((x + offset_x, y + offset_y,
                               x + w + offset_x, y + h + offset_y))

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for ti, track in enumerate(self.tracks):
            best_iou = self.min_iou
            best_di = -1
            for di, dbox in enumerate(det_bboxes):
                if di in matched_dets:
                    continue
                iou = self._iou(track.bbox, dbox)
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_di >= 0:
                matched_tracks.add(ti)
                matched_dets.add(best_di)
                track.bbox = det_bboxes[best_di]
                track.contour = contours[best_di]
                track.hits += 1
                track.misses = 0
                if track.hits >= self.min_hits:
                    track.confirmed = True

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1

        for di, (c, dbox) in enumerate(zip(contours, det_bboxes)):
            if di not in matched_dets:
                self.tracks.append(TrackedClutter(contour=c, bbox=dbox))

        self.tracks = [t for t in self.tracks if t.misses < self.max_misses]

    @property
    def confirmed_contours(self) -> list[np.ndarray]:
        return [t.contour for t in self.tracks if t.confirmed]

    @property
    def confirmed_count(self) -> int:
        return sum(1 for t in self.tracks if t.confirmed)


class DeskcleanViewport:
    """工位清洁视图，负责检测工具是否就位以及桌面清洁程度"""

    def __init__(self):
        self.latest_result = DeskcleanDetectResult()
        self._result_lock = threading.Lock()
        self._clutter_tracker = ClutterTracker()
        self.latest_frame_bgr: np.ndarray | None = None

        self._running = False
        self._yolo_thread: threading.Thread | None = None
        self._clutter_thread: threading.Thread | None = None

    def _detect_desk_clutter(self, frame: np.ndarray) -> DeskcleanDetectResult:

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (0.6, 0.5, 0.95, 0.95)  # 桌面区域 (相对坐标)
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        desk_img = frame[y1:y2, x1:x2]
        if desk_img.size == 0:
            return DeskcleanDetectResult()

        gray = cv2.cvtColor(desk_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_AREA = 120
        objects = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

        self._clutter_tracker.update(objects, x1, y1)

        clutter_mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask_region = clutter_mask[y1:y2, x1:x2]
        for c in self._clutter_tracker.confirmed_contours:
            cv2.drawContours(mask_region, [c], -1, [0, 0, 255], -1)

        return DeskcleanDetectResult(
            clutter_count=self._clutter_tracker.confirmed_count,
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
            frame = camera_service.get_frame(0)
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
            frame = camera_service.get_frame(0)
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
        frame = camera_service.get_frame(0)
        if frame is None:
            return

        result = deskclean_viewport.get_latest_result()
        tool_result = yolo_tools.latest_result
        overlay_frame = DeskcleanViewport._draw_overlay(frame, result, tool_result.boxes)
        present = set(tool_result.present)
        window.DeskcleanPageData.screwdriver_ready = "screwdriver" in present
        window.DeskcleanPageData.wire_stripper_ready = "wirestripper" in present
        window.DeskcleanPageData.multimeter_ready = "multimeter" in present
        window.DeskcleanPageData.crimping_ready = "crimping" in present

        # 算分
        TOOL_NAMES = {"screwdriver": "螺丝刀", "wirestripper": "剥线钳", "multimeter": "万用表", "crimping": "斜口钳"}
        all_tools = set(TOOL_NAMES.keys())
        missing_tools = all_tools - present

        TOOL_PENALTY = 10
        CLUTTER_PENALTY = 5
        MAX_CLUTTER_PENALTY = 100
        tool_deduction = len(missing_tools) * TOOL_PENALTY
        clutter_deduction = result.clutter_count * CLUTTER_PENALTY
        score = max(0, 100 - tool_deduction - clutter_deduction)

        lines = ["满分 100"]
        for t in missing_tools:
            lines.append(f"-{TOOL_PENALTY}（{TOOL_NAMES[t]}未归位）")
        if result.clutter_count > 0:
            lines.append(f"-{clutter_deduction}（{result.clutter_count} 个杂物，每个{CLUTTER_PENALTY}分）")
        window.DeskcleanPageData.clean_score = score
        if result.clutter_mask is not None and result.desk_region:
            x1, y1, x2, y2 = result.desk_region
            mask_pixels = np.count_nonzero(result.clutter_mask[y1:y2, x1:x2])
            total = (x2 - x1) * (y2 - y1) * 3
            window.DeskcleanPageData.clutter_mask_area = mask_pixels / total
        window.DeskcleanPageData.clean_description = "\n".join(lines)

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
                "clean_score": window.DeskcleanPageData.clean_score,
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
