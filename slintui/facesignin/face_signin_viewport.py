from dataclasses import dataclass, field
import threading
import time
from typing import List

import cv2
import numpy as np
import slint

from api_client import api_client
from camera_service import camera_service

TARGET_DETECTOR_MAX_DIM = 640


@dataclass
class FaceRecognizeResult:
    who: str = "识别中"
    faces: List[List[float]] = field(default_factory=list)
    aligned_bgr: np.ndarray | None = None
    frame_w: int = 0
    frame_h: int = 0


@dataclass
class PresenceState:
    name: str = "识别中"
    first_seen: float = 0.0
    uploaded: bool = False


class FaceSigninViewport:

    def __init__(self):
        self.latest_result = FaceRecognizeResult()
        self._result_lock = threading.Lock()
        self.latest_frame_bgr: np.ndarray | None = None

        self._running = False
        self._inference_thread: threading.Thread | None = None

        self._presence: PresenceState | None = None

        self.face_feats: np.ndarray | None = None
        self.face_names: np.ndarray | None = None

        self._detector_size = (640, 360)
        self._img_scale = 1.0
        self._last_cam_w = 0
        self._last_cam_h = 0

        self.detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar_int8bq.onnx", "",
            self._detector_size, score_threshold=0.7,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec_int8bq.onnx", "",
        )

    def _update_presence(self, who: str) -> None:
        now = time.monotonic()
        if who in ("", "识别中"):
            self._presence = None
            return

        if self._presence is None or self._presence.name != who:
            self._presence = PresenceState(name=who, first_seen=now, uploaded=False)

    def _reinit_detector(self, cam_w: int, cam_h: int) -> None:
        max_dim = max(cam_w, cam_h)
        self._img_scale = TARGET_DETECTOR_MAX_DIM / max_dim
        self._detector_size = (
            int(cam_w * self._img_scale),
            int(cam_h * self._img_scale),
        )
        self.detector.setInputSize(self._detector_size)

    def _postprocess(self, feat: np.ndarray) -> str:
        if self.face_feats is None:
            return "识别中"

        scores = np.dot(feat, self.face_feats.T)
        best_idx = np.argmax(scores)

        if scores[0, best_idx] < 40:
            return "识别中"

        return str(self.face_names[best_idx])

    @staticmethod
    def _draw_faces(overlay: np.ndarray, result: FaceRecognizeResult) -> None:
        h, w = overlay.shape[:2]
        sx = w / (result.frame_w or w) if result.frame_w else 1.0
        sy = h / (result.frame_h or h) if result.frame_h else 1.0

        for face_val in result.faces:
            x = int(face_val[0] * sx)
            y = int(face_val[1] * sy)
            bw = int(face_val[2] * sx)
            bh = int(face_val[3] * sy)
            reye_x = int(face_val[4] * sx)
            reye_y = int(face_val[5] * sy)
            leye_x = int(face_val[6] * sx)
            leye_y = int(face_val[7] * sy)
            nose_x = int(face_val[8] * sx)
            nose_y = int(face_val[9] * sy)
            rmouth_x = int(face_val[10] * sx)
            rmouth_y = int(face_val[11] * sy)
            lmouth_x = int(face_val[12] * sx)
            lmouth_y = int(face_val[13] * sy)

            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.drawMarker(overlay, (reye_x, reye_y), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(overlay, (leye_x, leye_y), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(overlay, (nose_x, nose_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(overlay, (rmouth_x, rmouth_y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.drawMarker(overlay, (lmouth_x, lmouth_y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

    def _inference_loop(self):
        print("[FaceSignin] 推理线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            if w != self._last_cam_w or h != self._last_cam_h:
                self._reinit_detector(w, h)
                self._last_cam_w = w
                self._last_cam_h = h

            t0 = time.perf_counter()

            scaled = cv2.resize(frame, self._detector_size, interpolation=cv2.INTER_LINEAR)
            _, faces = self.detector.detect(scaled)

            if faces is None:
                with self._result_lock:
                    self.latest_result = FaceRecognizeResult(
                        who="识别中", faces=[], aligned_bgr=None,
                        frame_w=w, frame_h=h,
                    )
                self._update_presence("识别中")
                continue

            t1 = time.perf_counter()

            best_face = max(faces, key=lambda face: face[14])
            aligned_bgr = self.recognizer.alignCrop(scaled, best_face)
            feat = self.recognizer.feature(aligned_bgr)

            t2 = time.perf_counter()
            who = self._postprocess(feat)
            t3 = time.perf_counter()

            print(
                f"[FaceSignin] {who} | detect={(t1 - t0) * 1000:.1f}ms "
                f"recog={(t2 - t1) * 1000:.1f}ms "
                f"postp={(t3 - t2) * 1000:.1f}ms"
            )

            faces_original = []
            for face in faces:
                face_orig = [v / self._img_scale for v in face[:14]] + [float(face[14])]
                faces_original.append(face_orig)

            result = FaceRecognizeResult(
                who=who, faces=faces_original, aligned_bgr=aligned_bgr,
                frame_w=w, frame_h=h,
            )
            with self._result_lock:
                self.latest_result = result
            self._update_presence(who)

        print("[FaceSignin] 推理线程退出")

    def start(self):
        if self._running:
            return
        self._running = True

        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        try:
            data = np.load("model_faces.npz", mmap_mode="r")
            self.face_feats = data["feats"]
            self.face_names = data["names"]
        except FileNotFoundError:
            print("[FaceSignin] 未找到人脸数据集 model_faces.npz，无法进行人脸签到")
            self.face_feats = None
            self.face_names = None

    def stop(self):
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)

    def get_latest_result(self) -> FaceRecognizeResult:
        with self._result_lock:
            return self.latest_result


face_signin_viewport = FaceSigninViewport()


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

    @slint.callback(global_name="FaceSigninPageData")
    async def request_signin_frame() -> None:
        result = face_signin_viewport.get_latest_result()
        window.FaceSigninPageData.signin_status_text = result.who

        now = time.monotonic()
        presence = face_signin_viewport._presence
        if presence is None:
            progress_pct = 0
        else:
            elapsed = max(0.0, now - presence.first_seen)
            progress_pct = int(min(100.0, (elapsed / 1.0) * 100.0))
        window.FaceSigninPageData.signin_progress_percent = progress_pct

        frame = camera_service.get_frame()
        if frame is None:
            return

        FaceSigninViewport._draw_faces(frame, result)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_signin_viewport.latest_frame_bgr = frame
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.FaceSigninPageData.camera_frame = slint.Image.load_from_array(arr)

        aligned = result.aligned_bgr
        if aligned is not None:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned_arr = np.ascontiguousarray(aligned_rgb, dtype=np.uint8)
            window.FaceSigninPageData.signin_aligned_frame = slint.Image.load_from_array(aligned_arr)

        if presence is not None and progress_pct >= 100 and not presence.uploaded:
            presence.uploaded = True
            await upload_current_face(presence.name, face_signin_viewport.latest_frame_bgr)

    window.FaceSigninPageData.request_signin_frame = request_signin_frame
    window.FaceSigninPageData.request_signin_frame()
