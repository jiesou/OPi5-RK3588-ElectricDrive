from inference_sdk import InferenceHTTPClient
import cv2
import time

# 初始化客户端
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="sD0Vt3yPtXgXajBC3sfT"
)

# 打开摄像头
cap = cv2.VideoCapture("rtsp://192.168.11.242:8554/test")

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("开始处理视频流...")

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
