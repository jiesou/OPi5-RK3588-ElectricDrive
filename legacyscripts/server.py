import cv2
import numpy as np
from flask import Flask, Response

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       display_width=640, display_height=640, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


app = Flask(__name__)

def generate():
    pipeline = gstreamer_pipeline(sensor_id=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    print(f"摄像头是否打开: {cap.isOpened()}")
    print(f"分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"格式: {cap.get(cv2.CAP_PROP_FOURCC)}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        frame_count += 1
        
        # 调试信息
        if frame_count % 30 == 0:  # 每秒打印一次
            print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}, "
                  f"min={frame.min()}, max={frame.max()}")
            
            # 检查是否全绿
            if frame.shape[2] == 3:
                b, g, r = cv2.split(frame)
                print(f"  通道均值: B={b.mean():.1f}, G={g.mean():.1f}, R={r.mean():.1f}")
        
        # 在画面上显示帧号
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # 编码
        ret_encode, buffer = cv2.imencode('.jpg', frame, 
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        if not ret_encode:
            print(f"Frame {frame_count}: 编码失败")
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>摄像头测试</title>
        <style>
            body { margin: 0; padding: 20px; background: #000; text-align: center; }
            img { max-width: 90%; border: 2px solid #fff; }
            h1 { color: #fff; }
        </style>
    </head>
    <body>
        <h1>摄像头原始输出测试</h1>
        <img src="/video" alt="Camera Feed">
        <p style="color: #fff;">如果看到红色画面显示 "CAMERA READ FAILED"，说明摄像头读取失败</p>
        <p style="color: #fff;">如果看到绿色画面，请检查终端输出的通道均值</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("=" * 60)
    print("摄像头测试服务器启动在: http://0.0.0.0:5000")
    print("请查看终端输出的调试信息")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
