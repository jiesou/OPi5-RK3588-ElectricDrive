import os
import cv2
import numpy as np
import threading
import queue
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# 设置 API key（可选：也可直接传入 init）
os.environ["ROBOFLOW_API_KEY"] = "sD0Vt3yPtXgXajBC3sfT"

# GStreamer 管道（Jetson Nano CSI 摄像头）
def gstreamer_pipeline(sensor_id=0, capture_width=1640, capture_height=1232,
                       display_width=640, display_height=640, framerate=20, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# 全局变量：线程安全的最新检测结果（用简单锁保护）
latest_detections = []
detections_lock = threading.Lock()

# ✅ 异步推理线程：从帧队列取最新帧推理
def inference_worker(frame_queue, model_id="electricdrive-yqqfl/8", api_key="sD0Vt3yPtXgXajBC3sfT"):
    global latest_detections
    # 创建本地推理 pipeline（注意：**不能复用主线程的 pipeline**）
    try:
        # 使用 InferencePipeline 但以 batch=False 的流式方式手动喂帧
        from inference import get_model
        model = get_model(model_id=model_id, api_key=api_key)

        print("[Inference Thread] Model loaded. Waiting for frames...")
        while True:
            try:
                # 清空队列，只取最新一帧（跳帧策略）
                frame = None
                while not frame_queue.empty():
                    frame = frame_queue.get_nowait()
                if frame is None:
                    continue

                # 推理（注意：model.infer() 输入是 list[np.ndarray]）
                results = model.infer([frame])[0]  # 取第一个（唯一）结果

                # 解析 detections
                detections = []
                if hasattr(results, 'predictions'):
                    for pred in results.predictions:
                        detections.append({
                            "x": pred.x,
                            "y": pred.y,
                            "width": pred.width,
                            "height": pred.height,
                            "class": pred.class_name,
                            "confidence": pred.confidence,
                        })

                # 更新全局 latest_detections（加锁）
                with detections_lock:
                    latest_detections = detections.copy()

                print(f"[Inference] Got {len(detections)} detections.")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Inference Thread] Error: {e}")
                import traceback; traceback.print_exc()
    except Exception as e:
        print(f"[Inference Thread Init] Failed: {e}")
        return


# ✅ 主显示循环：采集 + 显示（叠加最新检测）
def main_loop():
    global latest_detections

    # 初始化摄像头
    pipeline_str = gstreamer_pipeline(framerate=10)  # 采集10fps
    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("Failed to open GStreamer pipeline")

    print("[Main Loop] Camera opened. Starting inference thread...")

    # 帧队列（maxsize=1 实现“跳帧”：只保留最新帧待推理）
    frame_queue = queue.Queue(maxsize=1)

    # 启动推理线程
    inf_thread = threading.Thread(
        target=inference_worker,
        args=(frame_queue,),
        daemon=True  # 主线程退出时自动结束
    )
    inf_thread.start()

    print("[Main Loop] Started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 尝试将帧送入推理队列（非阻塞：若满则丢弃旧帧）
            try:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # 扔掉旧的
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(frame.copy())  # 送最新帧去推理
            except queue.Full:
                pass  # 极端情况：仍丢弃

            # 获取最新检测结果（加锁）
            with detections_lock:
                current_dets = latest_detections.copy()

            # 在当前帧上绘制最新检测框（可能滞后，但不卡）
            vis_frame = frame.copy()
            for det in current_dets:
                x, y = int(det["x"]), int(det["y"])
                w, h = int(det["width"]), int(det["height"])
                x1 = max(0, x - w // 2)
                y1 = max(0, y - h // 2)
                x2 = min(vis_frame.shape[1], x + w // 2)
                y2 = min(vis_frame.shape[0], y + h // 2)

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(vis_frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示
            cv2.imshow("Electric Drive Detection (10fps cam, async inference)", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
