import threading
import time
import cv2
import glob
import os
import re


def _get_usb_physical_port(dev_name):
    """返回 /dev/videoN 对应物理 USB 设备的端口标识（如 '2-1.3'），
       无法确定时返回 dev_name 自身作为兜底。"""
    dev_path = f"/sys/class/video4linux/{dev_name}/device"
    try:
        phys = os.path.realpath(dev_path)
    except OSError:
        return dev_name
    p = phys
    while p and p != "/":
        if os.path.isfile(os.path.join(p, "idVendor")) and os.path.isfile(os.path.join(p, "idProduct")):
            return os.path.basename(p)
        p = os.path.dirname(p)
    return dev_name


def _scan_cameras():
    """扫描摄像头，按 USB 物理端口去重，返回 [(device_path, (w, h)), ...]

    有些 USB 摄像头（如带激光雷达的 3D 相机）会创建多个 /dev/video* 节点，
    通过 sysfs 上的 USB 拓扑将同一物理设备的所有节点归为一组，每组只保留分辨率最高者。

    同一 USB 端口只测试第一个可用节点，避免频繁开/关多 video 节点导致摄像头固件异常。
    """
    grouped = {}
    seen_ports: set[str] = set()
    for dev in sorted(glob.glob("/dev/video*")):
        if not re.match(r'/dev/video\d+$', dev):
            continue
        dev_name = os.path.basename(dev)
        port = _get_usb_physical_port(dev_name)
        if port in seen_ports:
            continue
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                w, h = frame.shape[:2][::-1]
                seen_ports.add(port)
                grouped[port] = (dev, (w, h))
        else:
            cap.release()
    results = sorted(grouped.values(), key=lambda x: x[1][0] * x[1][1], reverse=True)
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
                if self._current_idx == 0:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
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
