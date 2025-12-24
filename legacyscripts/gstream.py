import os, sys
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'
import cv2
import time
def gstreamer_pipeline(sensor_id=0, capture_width=1640, capture_height=1232,
                       display_width=640, display_height=640, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )
    

pipeline = gstreamer_pipeline(sensor_id=0)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("开始处理视频流...")

from inference_sdk import InferenceHTTPClient

# 初始化客户端
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="sD0Vt3yPtXgXajBC3sfT"
)

print("初始化 client 完成")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 直接推理（不用 workflow）
        result = client.infer(
            frame,
            model_id="electricdrive-yqqfl/8"  # 你的模型 ID
        )
        
        # 处理结果
        print(f"检测到 {len(result.get('predictions', []))} 个对象")
        
        # 在图像上绘制结果
        for prediction in result.get('predictions', []):
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            confidence = prediction['confidence']
            class_name = prediction['class']
            
            # 画框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 画标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Inference', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 控制帧率（可选）
        time.sleep(0.033)  # ~30 FPS

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("视频流处理结束")
