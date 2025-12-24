import os
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
import slint

from camera_service import camera_service
from .signin_status_widget import append_log, init_signin_status, logs_model


class FaceSigninViewport:
    """独立的人脸签到视图，负责采集和基于 FaceRecognizerSF 的识别"""

    def __init__(self):

        self.latest_faces: list[Tuple[Tuple[int, int, int, int], str, float]] = []

        self._running = False
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None

        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar_int8bq.onnx", "", (640, 640))
        self.recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec_int8bq.onnx", "")
        self.gallery = self._load_gallery()
        self.match_threshold = 0.5

        self.latest_status_text = "等待签到"
        self.latest_person_text = ""
        self.latest_score = 0.0
        self._event_id = 0
        self._last_person = ""
        self._last_score = 0.0

    def _load_gallery(self) -> List[Tuple[str, np.ndarray]]:
        gallery_dir = "faces_gallery"
        if not os.path.isdir(gallery_dir):
            return []
        if self.detector is None or self.recognizer is None:
            return []

        features: List[Tuple[str, np.ndarray]] = []
        for fname in os.listdir(gallery_dir):
            path = os.path.join(gallery_dir, fname)
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            self.detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.detector.detect(img)
            if faces is None or len(faces) == 0:
                continue
            face = faces[0]
            aligned = self.recognizer.alignCrop(img, face)
            feat = self.recognizer.feature(aligned)
            name, _ = os.path.splitext(fname)
            features.append((name, feat))
        print(f"[FaceSignin] loaded gallery: {len(features)} entries")
        return features

    def _match(self, feat: np.ndarray) -> Tuple[str, float]:
        if not self.gallery:
            return "Unknown", 0.0
        best_name = "Unknown"
        best_score = -1.0
        for name, ref in self.gallery:
            score = float(self.recognizer.match(feat, ref, cv2.FaceRecognizerSF_FR_COSINE))
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.match_threshold:
            return "Unknown", best_score
        return best_name, best_score

    def _inference_loop(self):
        print("[FaceSignin] 推理线程启动")
        while self._running:
            frame = camera_service.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if self.detector is None or self.recognizer is None:
                with self._lock:
                    self.latest_faces = []
                    self.latest_status_text = "模型未就绪"
                    self.latest_person_text = ""
                    self.latest_score = 0.0
                time.sleep(0.05)
                continue

            self.detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = self.detector.detect(frame)

            faces_info: list[Tuple[Tuple[int, int, int, int], str, float]] = []
            best_name = ""
            best_score = 0.0

            if faces is not None:
                for face in faces:
                    x, y, w, h, score = face[:5]
                    if score < 0.4:
                        continue
                    box = (int(x), int(y), int(w), int(h))
                    aligned = self.recognizer.alignCrop(frame, face)
                    feat = self.recognizer.feature(aligned)
                    name, sim = self._match(feat)
                    best_name = name or "Unknown"
                    best_score = sim
                    faces_info.append((box, best_name, sim))

            status_text = "未检测到人脸" if best_name == "" else f"识别: {best_name} ({best_score:.2f})"
            if best_name == "":
                best_name = ""
                best_score = 0.0

            with self._lock:
                self.latest_faces = faces_info
                self.latest_status_text = status_text
                self.latest_person_text = best_name
                self.latest_score = best_score
                if best_name and (best_name != self._last_person or abs(best_score - self._last_score) > 1e-3):
                    self._event_id += 1
                    self._last_person = best_name
                    self._last_score = best_score
            time.sleep(0.005)
        print("[FaceSignin] 推理线程退出")

    def start(self):
        if self._running:
            return
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=1.0)


def bind_facesignin(window) -> None:
    init_signin_status(window)
    last_event = -1

    @slint.callback
    def request_signin_frame() -> None:
        nonlocal last_event
        # 拿原始帧
        frame = camera_service.get_frame()
        if frame is None:
            return
        with face_signin_viewport._lock:
            raw = frame.copy()
            faces = list(face_signin_viewport.latest_faces)
            status = face_signin_viewport.latest_status_text
            person = face_signin_viewport.latest_person_text
            score = face_signin_viewport.latest_score
            event_id = face_signin_viewport._event_id

        # 叠加框（可能相对稍滞后，但画面始终流畅）
        for (x, y, w, h), name, sim in faces:
            cv2.rectangle(raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                raw,
                f"{name} {sim:.2f}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.signin_frame = slint.Image.load_from_array(arr)

        window.signin_status_text = status
        window.signin_person_text = person
        window.signin_logs = logs_model

        if person and event_id != last_event:
            append_log(person, score)
            last_event = event_id

    window.request_signin_frame = request_signin_frame
    window.request_signin_frame()


face_signin_viewport = FaceSigninViewport()
