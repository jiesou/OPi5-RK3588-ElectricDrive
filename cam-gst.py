import cv2

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
else:
    print("摄像头已打开")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("帧读取失败")
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

cap.release()
cv2.destroyAllWindows()

