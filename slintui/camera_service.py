import threading
import time
import cv2
import glob
import numpy as np
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
    """摄像头服务，同时抓取所有可用摄像头的画面。"""

    def __init__(self):
        self._caps: dict[int, cv2.VideoCapture] = {}
        self._frames: dict[int, np.ndarray] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._running = False
        self._lock = threading.Lock()
        self.h, self.w = 720, 1280
        self._cameras = _scan_cameras()
        self._current_idx = 0
        print(f"[CameraService] 可用摄像头: {[c[0] for c in self._cameras]}")

    def start(self):
        if self._running or not self._cameras:
            return
        self._running = True
        for idx in range(len(self._cameras)):
            t = threading.Thread(target=self._capture_loop, args=(idx,), daemon=True)
            self._threads[idx] = t
            t.start()

    def _open_camera(self, idx):
        if idx >= len(self._cameras):
            return None
        dev_path = self._cameras[idx][0]
        for _ in range(3):
            cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            if cap.isOpened():
                if idx == 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                print(f"[CameraService] 打开摄像头 {idx}: {dev_path}")
                return cap
            cap.release()
            time.sleep(0.2)
        return None

    def _capture_loop(self, idx):
        cap = self._open_camera(idx)
        if cap is None:
            print(f"[CameraService] 无法打开摄像头 {idx}")
            return
        self._caps[idx] = cap
        while self._running:
            ok, frame = cap.read()
            if ok and frame is not None:
                if idx == 0:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self._lock:
                    self._frames[idx] = frame.copy()
                    self.h, self.w = frame.shape[:2]
            else:
                time.sleep(0.01)
        cap.release()
        self._caps.pop(idx, None)

    def get_frame(self, cam_id: int | None = None):
        if cam_id is None:
            cam_id = self._current_idx
        return self._frames.get(cam_id)

    def set_camera(self, cam_id: int):
        if cam_id < len(self._cameras):
            self._current_idx = cam_id

    @property
    def current_idx(self):
        return self._current_idx

    def stop(self):
        self._running = False
        for t in self._threads.values():
            t.join(timeout=1.0)
        for cap in self._caps.values():
            cap.release()


camera_service = CameraService()
