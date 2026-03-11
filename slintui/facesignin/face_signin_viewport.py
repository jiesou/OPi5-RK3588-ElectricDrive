from dataclasses import dataclass, field
import threading
import time
from typing import List

import cv2
from cv2.typing import MatLike
import numpy as np
import slint

from api_client import api_client
from camera_service import camera_service

# 等比例缩放，最长边控制在 640
IMG_SCALE = 640 / max(camera_service.h, camera_service.w)
IMG_SIZE = (int(camera_service.w * IMG_SCALE), int(camera_service.h * IMG_SCALE))

@dataclass
class FaceRecognizeResult:
    """完整的人脸识别结果"""
    who: str = "识别中"
    # Face[15]: x, y, w, h, 右眼x, y, 左眼x, y, 鼻尖x, y, 右嘴角x, y, 左嘴角x, y, 置信度 conf
    faces: List[MatLike] = field(default_factory=list)
    aligned_bgr: MatLike | None = None

@dataclass
class PresenceState:
    name: str = "识别中"
    first_seen: float = 0.0
    uploaded: bool = False

class FaceSigninViewport:
    """人脸签到视图，负责基于 cv2 的人脸识别"""

    def __init__(self):
        self.latest_result: FaceRecognizeResult = FaceRecognizeResult()
        self.latest_frame_bgr: np.ndarray | None = None

        self._running = False
        self._inference_thread: threading.Thread | None = None

        # 当前的驻留状态（None 表示没有人在驻留）
        self._presence: PresenceState | None = None

        # 人脸库向量与名字（在 start() 中加载）
        self.face_feats: np.ndarray | None = None
        self.face_names: np.ndarray | None = None

        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar_int8bq.onnx", "", IMG_SIZE, score_threshold=0.7)
        self.recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec_int8bq.onnx", "")

    def _update_presence(self, who: str) -> None:
        now = time.monotonic()
        # 当识别不到人或处于识别中状态时，清除驻留状态
        if who in ("", "识别中"):
            self._presence = None
            return

        # 如果当前没有驻留或识别到的是不同的人，重置驻留开始时间与上传标记
        if self._presence is None or self._presence.name != who:
            self._presence = PresenceState(name=who, first_seen=now, uploaded=False)
        # 如果是同一个人并且已经存在驻留记录，则保持 first_seen 不变（继续计时）

    def _preprocess(self, bgr: np.ndarray):
        return cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    
    def _postprocess(self, feat: np.ndarray) -> str:
        """返回识别到的人名"""
        # 如果没有加载人脸库，则返回 识别中
        if self.face_feats is None:
            return "识别中"

        # 使用矩阵运算计算所有人脸的余弦相似度
        # 这样可以一次性通过向量化计算加速最近邻查找
        scores = np.dot(feat, self.face_feats.T)
        best_idx = np.argmax(scores)

        # 相似度阈值可以高一点，更加金融级安全
        if scores[0, best_idx] < 40:
            return "识别中"

        name = self.face_names[best_idx]
        return str(name)

    def _inference_loop(self):
        print("[FaceSignin] 推理线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            t_detect_start = time.perf_counter()

            aligned_bgr = None

            frame = self._preprocess(frame)
            _, faces = self.detector.detect(frame)
            if faces is None:
                self.latest_result = FaceRecognizeResult(
                    who="识别中",
                    faces=[],
                    aligned_bgr=None
                )
                self._update_presence("识别中")
                continue

            t_detect_end = time.perf_counter()
            t_recognize_start = time.perf_counter()

            # 找到最像人脸的人脸
            best_face = max(faces, key=lambda face: face[14])
            # 对齐、识别
            aligned_bgr = self.recognizer.alignCrop(frame, best_face)
            feat = self.recognizer.feature(aligned_bgr)

            t_recognize_end = time.perf_counter()
            t_postprocess_start = time.perf_counter()
            who = self._postprocess(feat)
            print(f"[FaceSignin] 识别到人脸: {who}")
            t_postprocess_end = time.perf_counter()

            print(f"[FaceSignin] Timing (ms): detect={(t_detect_end - t_detect_start)*1000:.1f} recognize={(t_recognize_end - t_recognize_start)*1000:.1f} postprocess={(t_postprocess_end - t_postprocess_start)*1000:.1f} total={(t_postprocess_end - t_detect_start)*1000:.1f}")

            self.latest_result = FaceRecognizeResult(
                who=who,
                faces=faces,
                aligned_bgr=aligned_bgr
            )
            self._update_presence(who)

        print("[FaceSignin] 推理线程退出")

    def start(self):
        if self._running:
            return
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        try:
            data = np.load("model_faces.npz", mmap_mode='r')
            # 生成脚本保存时使用 keys: feats, names
            self.face_feats = data['feats']
            self.face_names = data['names']
        except FileNotFoundError:
            print("[FaceSignin] 未找到人脸数据集 model_faces.npz，无法进行人脸签到")
            self.face_feats = None
            self.face_names = None

    def stop(self):
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)


def bind_facesignin(window) -> None:
    async def upload_current_face(name: str, frame_bgr: np.ndarray) -> None:
        try:
            ok, buffer = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                raise RuntimeError("编码 JPEG 失败")

            response = await api_client.upload_face_async(buffer.tobytes(), name)

            if not response.get("success"):
                error = response.get("error", "未知")
                print(f"[FaceSignin] 上传失败: {error}")
                window.show_temporary_message(f"你好，{name}: {error}")
                return

            window.show_temporary_message(f"你好，{name}")
        except Exception as e:
            print(f"[FaceSignin] 上传失败: {e}")
            window.show_temporary_message(f"你好，{name}: {e}")

    @slint.callback
    async def request_signin_frame() -> None:
        who = face_signin_viewport.latest_result.who
        window.FaceSigninPageData.signin_status_text = who
        # 更新人脸滞留等待状态
        now = time.monotonic()
        if face_signin_viewport._presence is None:
            progress_pct = 0
            wait_seconds_needed = 1.0
        else:
            elapsed = max(0.0, now - face_signin_viewport._presence.first_seen)
            wait_seconds_needed = 1.0
            progress_pct = int(min(100.0, (elapsed / wait_seconds_needed) * 100.0))
        window.FaceSigninPageData.signin_progress_percent = progress_pct

        # 拿原始帧
        frame = camera_service.get_frame()
        if frame is None:
            return
         
        # 叠加框
        for face_val in face_signin_viewport.latest_result.faces:
            # 将所有坐标除以 scale 还原到原始大图尺寸
            f = [int(val / IMG_SCALE) for val in face_val]
            x, y, w, h = f[0:4]
            reye_x, reye_y = f[4:6]
            leye_x, leye_y = f[6:8]
            nose_x, nose_y = f[8:10]
            rmouth_x, rmouth_y = f[10:12]
            lmouth_x, lmouth_y = f[12:14]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawMarker(frame, (reye_x, reye_y), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(frame, (leye_x, leye_y), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(frame, (nose_x, nose_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(frame, (rmouth_x, rmouth_y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(frame, (lmouth_x, lmouth_y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_signin_viewport.latest_frame_bgr = frame
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.FaceSigninPageData.camera_frame = slint.Image.load_from_array(arr)

        aligned = face_signin_viewport.latest_result.aligned_bgr
        if aligned is not None:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned_arr = np.ascontiguousarray(aligned_rgb, dtype=np.uint8)
            window.FaceSigninPageData.signin_aligned_frame = slint.Image.load_from_array(aligned_arr)

        # 人脸驻留进度走满，则触发上传（确保 _presence 存在）
        if face_signin_viewport._presence is not None and progress_pct >= 100 and not face_signin_viewport._presence.uploaded:
            face_signin_viewport._presence.uploaded = True
            await upload_current_face(face_signin_viewport._presence.name, face_signin_viewport.latest_frame_bgr)

    window.FaceSigninPageData.request_signin_frame = request_signin_frame
    window.FaceSigninPageData.request_signin_frame()

face_signin_viewport = FaceSigninViewport()
