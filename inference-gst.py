import os
import cv2
import numpy as np

# 设置 API key
os.environ["ROBOFLOW_API_KEY"] = "sD0Vt3yPtXgXajBC3sfT"

# 配置 GStreamer 管道
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


from inference.core.interfaces.camera.entities import VideoFrame
# 自定义 sink 函数：在帧上绘制检测框并显示
def custom_sink(predictions: dict, video_frame: VideoFrame):
    # 复制原始图像以避免修改原始数据
    image = video_frame.image.copy()
    
    # 检查是否有预测结果
    if "predictions" in predictions and predictions["predictions"]:
        # 遍历所有预测
        for pred in predictions["predictions"]:
            # 获取边界框坐标
            x = int(pred["x"])
            y = int(pred["y"])
            width = int(pred["width"])
            height = int(pred["height"])
            
            # 计算边界框的四个角点
            x1 = max(0, x - width // 2)
            y1 = max(0, y - height // 2)
            x2 = min(image.shape[1], x + width // 2)
            y2 = min(image.shape[0], y + height // 2)
            
            # 绘制边界框 (BGR 格式)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{pred['class']} {pred['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示处理后的帧
    cv2.imshow("Electric Drive Detection", image)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    
    return True

# 创建 GStreamer 管道
pipeline_str = gstreamer_pipeline(sensor_id=0, flip_method=2)  # flip_method=2 通常适用于 Jetson Nano 摄像头

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes

# 初始化推理管道
try:
    pipeline = InferencePipeline.init(
        model_id="electricdrive-yqqfl/8",  # 你的模型 ID
        video_reference=pipeline_str,      # GStreamer 管道
        api_key="sD0Vt3yPtXgXajBC3sfT",    # API key
        on_prediction=custom_sink,         # 自定义处理函数
    )
    
    print("开始处理视频流... 按 'q' 键退出")
    print(f"GStreamer 管道: {pipeline_str}")
    
    # 启动管道
    pipeline.start()
    pipeline.join()

except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
