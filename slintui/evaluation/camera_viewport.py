import threading
import time
import cv2
import slint
import numpy as np

from camera_service import camera_service
from .yolo import yolo


class CameraViewport:
    """
    相机视图端口，负责图像采集和显示
    
    优先级：
    1. 如果 ./test_image.jpg 存在，使用测试图片
    2. 否则尝试打开摄像头设备 0
    
    后台线程持续运行：
    - 采集线程：持续捕获摄像头帧
    - 推理线程：持续对最新帧进行 YOLO 推理
    - UI 线程：定时读取处理好的帧显示
    """

    def __init__(self):
        self._test_frame_bgr: np.ndarray | None = None

        # 显示帧（BGR已画框）
        self.latest_frame_bgr: np.ndarray | None = None
        
        # 线程控制
        self._running = False
        self.inference_enabled = False
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None

    def _inference_loop(self):
        """后台线程：持续对最新帧进行 YOLO 推理并叠加绘制"""
        print("[CameraViewport] 推理线程启动")

        while self._running:
            frame = camera_service.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                yolo.detect(frame)

            # 小的等待以避免占用 100% CPU
            time.sleep(0.001)

        print("[CameraViewport] 推理线程退出")

    def start(self):
        """启动后台线程"""
        if self._running:
            return
        
        self._running = True
        # 启动推理线程
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)



# 全局单例
camera_viewport = CameraViewport()


def bind_camera(window) -> None:
    """绑定相机采集逻辑到 Slint 窗口（不自动启动线程）"""

    # 会被 UI 定时调用
    @slint.callback
    def request_camera_frame() -> None:
        # 同步推理开关状态到后台线程
        camera_viewport.inference_enabled = bool(window.inference_enabled)
        
        # 读取已处理好的帧
        frame = camera_service.get_frame()
        if frame is None:
            return
        drawn_frame = frame.copy()
            
        # 如果推理未启用，直接把原始帧作为显示帧
        if not camera_viewport.inference_enabled:
            pass
        else:
            result = yolo.latest_result
            for box in result.boxes:
                if box.label == "terminal":
                    box_color = (0, 255, 0)
                elif box.label == "cross":
                    box_color = (0, 0, 255)
                elif box.label == "excopper":
                    box_color = (0, 255, 255)
                elif box.label == "exterminal":
                    box_color = (255, 0, 0)
                if box.source == 0:
                    label_color = (255, 0, 255)
                elif box.source == 1:
                    label_color = (0, 255, 255)
                elif box.source == 2:
                    label_color = (255, 255, 255)
                cv2.rectangle(drawn_frame, (box.x1, box.y1), (box.x2, box.y2), box_color, 2)
                cv2.putText(
                    drawn_frame,
                    f"{box.label} {box.conf:.2f}",
                    (box.x1, max(0, box.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color,
                    2,
                    cv2.LINE_AA,
                )

        # 创建 Slint 图像
        rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
        camera_viewport.latest_frame_bgr = drawn_frame
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.camera_frame = slint.Image.load_from_array(arr)
        
        # 获取最新检测结果用于显示
        detection = yolo.latest_result.detection
        
        window.current_detection_text = (
            f"当前: 号码管={detection.terminal} 交叉={detection.cross} "
            f"露铜={detection.excopper} 露端={detection.exterminal}"
        )

    window.request_camera_frame = request_camera_frame
