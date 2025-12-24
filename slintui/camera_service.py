import threading
import time
import cv2


class CameraService:
    """单摄像头采集服务，负责持续抓取最新帧供各视图复用"""

    def __init__(self):
        self._cap = None
        self._frame = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        self._cap = cv2.VideoCapture(0)
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame.copy()
            else:
                time.sleep(0.01)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self):
        return self._frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


camera_service = CameraService()
