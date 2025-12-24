from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import time

import cv2
import numpy as np

IMG_SIZE = (640, 640)

_thread_local = threading.local()

def _get_models():
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar_int8bq.onnx", "", IMG_SIZE)
        _thread_local.recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec_int8bq.onnx", "")
    return _thread_local.detector, _thread_local.recognizer

def _preprocess(bgr: np.ndarray):
    return cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

def _job(image_path: Path):
    detector, recognizer = _get_models()
    bgr = cv2.imread(str(image_path))
    frame = _preprocess(bgr)
    _, faces = detector.detect(frame)
    if faces is None:
        print(f"[generate] WARN 未检测到人脸: {image_path.name}")
    best_face = max(faces, key=lambda face: face[14]) # face[14] -> conf
    aligned = recognizer.alignCrop(frame, best_face)
    feat = recognizer.feature(aligned).flatten()
    # 文件名（不含扩展名）作为人名
    return image_path.stem, feat


def main():
    images = [p for p in Path("dataset_face").iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]
    total = len(images)
    names: list[str] = []
    feats: list[np.ndarray] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_job, p): p for p in images}
        for idx, future in enumerate(as_completed(futures), start=1):
            name, feat = future.result()
            names.append(name)
            feats.append(feat)
            print(f"[generate] {idx}/{total} processed: {name}")

    # 一维向量转二维矩阵，便于后期处理
    feats_matrix = np.stack(feats, axis=0)
    names_array = np.array(names)
    t_end = time.perf_counter()

    np.savez("model_faces.npz", feats=feats_matrix, names=names_array)
    print(f"[generate] saved {len(names)} faces")
    print(f"[generate] {(t_end - t_start)*1000.0/len(names):.2f} ms per face")


if __name__ == "__main__":
    main()
