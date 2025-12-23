import os
import threading
import time
import cv2
import slint
import numpy as np

from yolo import yolo


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
        self._cap = None
        self._test_frame_bgr: np.ndarray | None = None

        # 原始帧（BGR未画框）
        self._raw_frame_bgr: np.ndarray | None = None
        # 显示帧（BGR已画框）
        self.latest_frame_bgr: np.ndarray | None = None
        
        # 线程控制
        self._running = False
        self.inference_enabled = False
        self._lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        
    def _ensure_capture(self):
        """确保摄像头已打开"""
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        # 如果设备文件不存在，直接返回 None，避免重复尝试打开并产生 V4L 警告
        if not os.path.exists('/dev/video0'):
            return None

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
        return self._cap

    def _capture_loop(self):
        """后台线程：持续捕获摄像头帧"""
        print("[CameraViewport] 采集线程启动")
        
        while self._running:
            cap = self._ensure_capture()
            if cap is None:
                time.sleep(0.1)
                no_camera_img = cv2.imread("no_camera.jpg")
                if no_camera_img is not None:
                    self._raw_frame_bgr = no_camera_img.copy()
                continue
            
            ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._raw_frame_bgr = frame.copy()
            else:
                print("[CameraViewport] 采集失败，重试中...")
                time.sleep(0.01)
        
        print("[CameraViewport] 采集线程退出")

    def _inference_loop(self):
        """后台线程：持续对最新帧进行 YOLO 推理并叠加绘制"""
        print("[CameraViewport] 推理线程启动")

        while self._running:
            # 获取当前帧的副本用于推理
            if self._raw_frame_bgr is None:
                time.sleep(0.01)
                continue
            with self._lock:
                frame = self._raw_frame_bgr.copy()
            try:
                yolo.detect(frame)
            except Exception:
                time.sleep(0.01)

            # 小的等待以避免占用 100% CPU
            time.sleep(0.001)

        print("[CameraViewport] 推理线程退出")

    def start(self):
        """启动后台线程"""
        if self._running:
            return
        
        self._running = True
        
        # 启动采集线程
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        # 启动推理线程
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        """停止后台线程"""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)

    def close(self) -> None:
        """关闭摄像头"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# 全局单例
camera_viewport = CameraViewport()


def bind_camera(window) -> None:
    """绑定相机采集逻辑到 Slint 窗口"""
    camera_viewport.start()

    # 会被 UI 定时调用
    @slint.callback
    def request_camera_frame() -> None:
        # 同步推理开关状态到后台线程
        camera_viewport.inference_enabled = bool(window.inference_enabled)
        
        # 读取已处理好的帧
        if camera_viewport._raw_frame_bgr is None:
            return
        with camera_viewport._lock:
            drawn_frame = camera_viewport._raw_frame_bgr.copy()
            
        # 如果推理未启用，直接把原始帧作为显示帧
        if not camera_viewport.inference_enabled:
            pass
        else:
            result = yolo.latest_result
            for box in result.boxes:
                if box.label == "terminal":
                    color = (0, 255, 0)
                elif box.label == "cross":
                    color = (255, 0, 0)
                elif box.label == "excopper":
                    color = (255, 255, 0)
                elif box.label == "exterminal":
                    color = (0, 0, 255)
                cv2.rectangle(drawn_frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                cv2.putText(
                    drawn_frame,
                    f"{box.label} {box.conf:.2f}",
                    (box.x1, max(0, box.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
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
    window.request_camera_frame() # 初始调用
