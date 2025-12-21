import os
from pathlib import Path
import time

import cv2
import slint

import numpy as np

from yolo import yolo


class CameraViewport:
    """Owns camera frame acquisition.

    For now:
    - if ./test_image.jpg exists, use it (repeat)
    - otherwise try to read from camera device 0
    """

    def __init__(self):
        self._cap = None
        self._test_image_path: str | None = None
        self._test_frame_bgr: np.ndarray | None = None

        self._frame_dir = Path("/tmp/electricdrive")
        self._frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_toggle = 0

        self._latest_frame_bgr: np.ndarray | None = None
        self._last_det_counts = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}
        self._last_det_boxes = []
        self._last_det_at = 0.0

        test_path = Path(os.getcwd()) / "test_image.jpg"
        if test_path.exists():
            self._test_image_path = str(test_path)
            img = cv2.imread(self._test_image_path)
            if img is not None:
                self._test_frame_bgr = img
        else:
            # Generate a small test frame so UI always has something to show.
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            cv2.putText(
                frame,
                "No camera / no test_image.jpg",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            self._test_frame_bgr = frame

    def _write_jpeg(self, bgr_frame: np.ndarray) -> str | None:
        # Alternate between two paths so that the UI sees a change.
        name = f"frame_{self._frame_toggle}.jpg"
        self._frame_toggle = 1 - self._frame_toggle
        path = str(self._frame_dir / name)
        ok = cv2.imwrite(path, bgr_frame)
        return path if ok else None

    def _ensure_capture(self):
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        # In the future we can use stored_settings.get("image_feed_udp_url")
        # to open UDP / gstreamer pipelines. Keep it minimal for now.
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
        return self._cap

    def get_latest_frame(self) -> np.ndarray | None:
        return self._latest_frame_bgr

    def _overlay(self, bgr_frame: np.ndarray) -> np.ndarray:
        # Draw boxes and summary on a copy.
        out = bgr_frame.copy()

        for box in self._last_det_boxes:
            x1, y1, x2, y2, label, conf = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        c = self._last_det_counts
        cv2.putText(
            out,
            f"terminal={c['terminal']} cross={c['cross']} excopper={c['excopper']} exterminal={c['exterminal']}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _maybe_run_inference(self, frame: np.ndarray) -> None:
        # Rate-limit inference so the UI stays responsive.
        now = time.monotonic()
        if now - self._last_det_at < 0.5:
            return

        det = yolo.detect(frame)
        self._last_det_counts = dict(det.counts)
        self._last_det_boxes = [
            (b.x1, b.y1, b.x2, b.y2, b.label, b.conf) for b in det.boxes
        ]
        self._last_det_at = now

    def read_image(self, inference_enabled: bool):
        """Return (slint.Image, counts_dict).

        Important: In this environment slint.Image is builtins.PyImage and
        does NOT provide load_from_array(). So we update the UI via
        Image.load_from_path() with a temp jpeg file.
        """

        if self._test_frame_bgr is not None:
            frame = self._test_frame_bgr
        else:
            cap = self._ensure_capture()
            if cap is None:
                return None, self._last_det_counts
            ok, frame = cap.read()
            if not ok or frame is None:
                return None, self._last_det_counts

        self._latest_frame_bgr = frame

        if inference_enabled:
            self._maybe_run_inference(frame)

        drawn = self._overlay(frame) if inference_enabled else frame

        path = self._write_jpeg(drawn)
        if path is None:
            return None, self._last_det_counts

        return slint.Image.load_from_path(path), self._last_det_counts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


camera_viewport = CameraViewport()


def bind_camera(window) -> None:
    """Bind camera acquisition to the Slint window."""

    def request_camera_frame() -> None:
        img, counts = camera_viewport.read_image(bool(window.inference_enabled))
        if img is None:
            return
        window.camera_frame = img
        window.current_detection_text = (
            f"当前: 号码管={counts['terminal']} 交叉={counts['cross']} 露铜={counts['excopper']} 露端={counts['exterminal']}"
        )

    window.request_camera_frame = request_camera_frame
