import threading
import time
import cv2


class CameraService:
    """摄像头采集服务，负责持续抓取最新帧供各视图复用"""

    def __init__(self):
        self._caps = [None, None]  # 两个摄像头
        self._frames = [None, None]  # 两个帧缓存
        self._running = False
        self._threads: list[threading.Thread | None] = [None, None]
        self._locks = [threading.Lock(), threading.Lock()]
        self.h = 720
        self.w = 1280
        self._swapped = False  # 是否交换了摄像头映射

    def start(self):
        if self._running:
            return
        self._running = True
        for i in range(2):
            self._threads[i] = threading.Thread(target=self._capture_loop, args=(i,), daemon=True)
            self._threads[i].start()

    def _capture_loop(self, cam_id: int):
        self._caps[cam_id] = cv2.VideoCapture(cam_id)
        while self._running:
            ok, frame = self._caps[cam_id].read()
            if ok and frame is not None:
                with self._locks[cam_id]:
                    self._frames[cam_id] = frame.copy()
                    if cam_id == 0:
                        self.h, self.w = frame.shape[:2]
            else:
                time.sleep(0.01)
        if self._caps[cam_id] is not None:
            self._caps[cam_id].release()
            self._caps[cam_id] = None

    def get_frame(self, cam_id: int = 0):
        """获取帧，cam_id=0 为主摄像头，cam_id=1 为副摄像头（人脸识别专用）
        单摄像头环境下副摄像头会 fallback 到主摄像头
        """
        # 如果交换了，主摄像头实际取物理摄像头1，副摄像头取物理摄像头0
        physical_id = 1 - cam_id if self._swapped else cam_id
        frame = self._frames[physical_id]
        # 单摄像头 fallback：副摄像头无帧时使用主摄像头
        if frame is None and cam_id == 1:
            frame = self._frames[0 if not self._swapped else 1]
        return frame

    def swap_cameras(self):
        """交换主/副摄像头的映射"""
        self._swapped = not self._swapped

    @property
    def swapped(self):
        return self._swapped

    def stop(self):
        self._running = False
        for i in range(2):
            if self._threads[i]:
                self._threads[i].join(timeout=1.0)
            if self._caps[i] is not None:
                self._caps[i].release()
                self._caps[i] = None


camera_service = CameraService()
