import os
from pathlib import Path
import time

import cv2
import slint
import numpy as np

from yolo import yolo, Detection


class CameraViewport:
    """
    相机视图端口，负责图像采集和显示
    
    优先级：
    1. 如果 ./test_image.jpg 存在，使用测试图片
    2. 否则尝试打开摄像头设备 0
    """

    def __init__(self):
        self._cap = None
        self._test_image_path: str | None = None
        self._test_frame_bgr: np.ndarray | None = None

        self._frame_dir = Path("/tmp/electricdrive")
        self._frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_toggle = 0

        self._latest_frame_bgr: np.ndarray | None = None

        # 检查是否存在测试图片
        test_path = Path(os.getcwd()) / "test_image.jpg"
        if test_path.exists():
            self._test_image_path = str(test_path)
            img = cv2.imread(self._test_image_path)
            if img is not None:
                self._test_frame_bgr = img
        else:
            # 生成一个占位帧，确保 UI 始终有内容显示
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            cv2.putText(
                frame,
                "No camera / no test_image.jpg",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            self._test_frame_bgr = frame

    def _write_jpeg(self, bgr_frame: np.ndarray) -> str | None:
        """
        将帧写入临时 JPEG 文件
        在两个文件名之间交替，确保 UI 能检测到变化
        """
        name = f"frame_{self._frame_toggle}.jpg"
        self._frame_toggle = 1 - self._frame_toggle
        path = str(self._frame_dir / name)
        ok = cv2.imwrite(path, bgr_frame)
        return path if ok else None

    def _ensure_capture(self):
        """确保摄像头已打开"""
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        # 未来可以使用 stored_settings.get("image_feed_udp_url")
        # 来打开 UDP / GStreamer 管道
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
        return self._cap

    def get_latest_frame(self) -> np.ndarray | None:
        """获取最新的原始帧（用于拍摄）"""
        return self._latest_frame_bgr

    def _overlay_detections(self, bgr_frame: np.ndarray, result) -> np.ndarray:
        """
        在帧上绘制检测框和统计信息
        
        Args:
            bgr_frame: 原始 BGR 帧
            result: YoloResult 对象
            
        Returns:
            绘制了标注的帧副本
        """
        out = bgr_frame.copy()

        # 绘制检测框
        for box in result.boxes:
            cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{box.label} {box.conf:.2f}",
                (box.x1, max(0, box.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 绘制统计信息
        d = result.detection
        cv2.putText(
            out,
            f"terminal={d.terminal} cross={d.cross} excopper={d.excopper} exterminal={d.exterminal}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def read_image(self, inference_enabled: bool) -> slint.Image | None:
        """
        读取并处理图像帧
        
        Args:
            inference_enabled: 是否启用推理
            
        Returns:
            (slint.Image, Detection): 图像对象和检测结果
        """
        # 获取原始帧
        if self._test_frame_bgr is not None:
            frame = self._test_frame_bgr
        else:
            cap = self._ensure_capture()
            if cap is None:
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None

        self._latest_frame_bgr = frame

        # 如果启用推理，运行检测
        if inference_enabled:
            yolo.detect(frame)

        # 获取最新的检测结果
        result = yolo.latest_result

        # 绘制标注（仅在启用推理时）
        drawn = self._overlay_detections(frame, result) if inference_enabled else frame

        # 写入临时文件并加载为 Slint 图像
        path = self._write_jpeg(drawn)
        if path is None:
            return None

        return slint.Image.load_from_path(path)

    def close(self) -> None:
        """关闭摄像头"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# 全局单例
camera_viewport = CameraViewport()


def bind_camera(window) -> None:
    """绑定相机采集逻辑到 Slint 窗口"""

    # 会被 UI 定时调用
    def request_camera_frame() -> None:
        img = camera_viewport.read_image(bool(window.inference_enabled))
        detection = yolo.latest_result.detection
        if img is None:
            return
        
        window.camera_frame = img
        window.current_detection_text = (
            f"当前: 号码管={detection.terminal} 交叉={detection.cross} "
            f"露铜={detection.excopper} 露端={detection.exterminal}"
        )

    window.request_camera_frame = request_camera_frame
