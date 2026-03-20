import threading
import time
import cv2
import glob


def _scan_cameras():
    """扫描摄像头，按分辨率降序返回 [(device_path, (w, h)), ...]"""
    results = []
    for dev in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                results.append((dev, frame.shape[:2][::-1]))
            cap.release()
    results.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)
    return results


class CameraService:
    """摄像头服务，同一时间只能使用一个摄像头（USB带宽限制），支持切换。"""

    def __init__(self):
        self._cap = None
        self._frame = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.h, self.w = 720, 1280
        self._cameras = _scan_cameras()
        self._current_idx = 0
        self._target_idx = 0
        print(f"[CameraService] 可用摄像头: {[c[0] for c in self._cameras]}")

    def start(self):
        if self._running or not self._cameras:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open_camera(self, idx):
        if idx >= len(self._cameras):
            return None
        dev_path = self._cameras[idx][0]
        for _ in range(3):
            cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"[CameraService] 打开摄像头 {idx}: {dev_path}")
                return cap
            cap.release()
            time.sleep(0.2)
        return None

    def _capture_loop(self):
        self._cap = self._open_camera(self._current_idx)
        while self._running:
            with self._lock:
                target_idx = self._target_idx
            if target_idx != self._current_idx:
                if self._cap:
                    self._cap.release()
                    self._cap = None
                time.sleep(0.2)
                self._cap = self._open_camera(target_idx)
                self._current_idx = target_idx
                self._frame = None
            if self._cap is None:
                time.sleep(0.01)
                continue
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._frame = frame.copy()
                self.h, self.w = frame.shape[:2]
            else:
                time.sleep(0.01)
        if self._cap:
            self._cap.release()

    def get_frame(self, cam_id: int = 0):
        return self._frame

    def set_camera(self, cam_id: int):
        if cam_id < len(self._cameras) and cam_id != self._target_idx:
            with self._lock:
                self._target_idx = cam_id

    @property
    def current_idx(self):
        return self._current_idx

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


camera_service = CameraService()
