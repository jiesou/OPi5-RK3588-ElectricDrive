import cv2
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QMutex
from PySide6.QtGui import QImage, QPixmap

from .state_bus import stateBus, Detection

from ultralytics import YOLO

FRAMERATE = 30

# 每类的 BGR 颜色映射（OpenCV 使用 BGR）
CLASS_COLORS = {
    "terminal": (0, 255, 0),     # 绿色
    "exterminal": (255, 0, 0), # 蓝色
    "excopper": (0, 255, 255),     # 黄色
    "cross": (0, 0, 255),        # 红色
}

def gstreamer_pipeline(sensor_id=0, capture_width=1640, capture_height=1232,
                       display_width=640, display_height=640, framerate=FRAMERATE, flip_method=1):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


class InferenceThread(QThread):
    """YOLO 推理线程"""
    results_ready = Signal(list)  # 发送检测结果
    
    def __init__(self):
        super().__init__()
        self.model = None
        self._running = True
        self._frame = None
        self._frame_mutex = QMutex()
        
    def run(self):
        """线程主循环"""
        if YOLO is None:
            print("[InferenceThread] YOLO not available")
            return
            
        try:
            self.model = YOLO("electricdrivev10.3.15.2_rknn_model", task='detect')
            print(f"[InferenceThread] Model loaded")
        except Exception as e:
            print(f"[InferenceThread] Failed to load model: {e}")
            return
        
        while self._running:
            if not stateBus.get_inference_enabled():
                self.msleep(100) # 开关没开，就睡 0.1 秒，不干活
                continue
            self._frame_mutex.lock()
            frame = self._frame
            self._frame = None
            self._frame_mutex.unlock()
            
            if frame is None:
                self.msleep(50)
                continue
            
            # try:
            yolo_result = self.model.predict(frame,
                conf=0.05,   # 保留置信度 ≥ 0.05 的检测框
                iou=0.2,    # NMS 的 IoU 阈值设为 0.3，不那么容易重叠
                verbose=True
            )
            results = []
            for result in yolo_result:
                boxes = result.boxes
                for box in boxes:
                    class_name = result.names[int(box.cls[0])]
                    if class_name == "class3":
                        class_name = "terminal"
                    elif class_name == "class2":
                        class_name = "exterminal"
                    elif class_name == "class1":
                        class_name = "excopper"
                    elif class_name == "class0":
                        class_name = "cross"
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    results.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "class": class_name,
                        "confidence": float(box.conf[0]),
                    })
            self.results_ready.emit(results)
            # except Exception as e:
            #     print(f"[InferenceThread] Error: {e}")
                
    def set_frame(self, frame):
        """设置待推理的帧"""
        self._frame_mutex.lock()
        # 避免重复拷贝：如果已有待处理帧，跳过本次
        if self._frame is None:
            self._frame = frame.copy()
        self._frame_mutex.unlock()
        
    def stop(self):
        self._running = False
        self.wait()


class CameraViewport(QWidget):
    """摄像头视图，带 YOLO 推理"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 图形视图
        self.view = QGraphicsView()
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignCenter)
        self.scene = QGraphicsScene(self)
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)
        self.view.setScene(self.scene)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        # 摄像头
        self.cap = None
        self._init_camera()
        
        # 最新的检测结果
        self._results = []
        
        # 推理线程
        self.inference_thread = InferenceThread()
        self.inference_thread.results_ready.connect(self._on_results_ready)
        # 根据初始状态决定是否启动
        if stateBus.get_inference_enabled():
            self.inference_thread.start()
        
        # 定时器：采集和显示帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(1000 // FRAMERATE)

        stateBus.inference_enabled_changed.connect(self._on_inference_enabled_changed)
        
    def _on_results_ready(self, results):
        """接收推理结果的槽函数"""
        self._results = results
        terminal_num = 0
        cross_num = 0
        excopper_num = 0
        exterminal_num = 0
        for det in results:
            class_name = det["class"]
            if class_name == "terminal":
                terminal_num += 1
            elif class_name == "cross":
                cross_num += 1
            elif class_name == "excopper":
                excopper_num += 1
            elif class_name == "exterminal":
                exterminal_num += 1
                
        stateBus.set_detections(Detection(
            terminal=terminal_num,
            cross=cross_num,
            excopper=excopper_num,
            exterminal=exterminal_num
        ))

    def _init_camera(self):
        """初始化摄像头"""
        # pipeline = gstreamer_pipeline(framerate=FRAMERATE)
        # self.cap = cv2.VideoCapture(0)
        # if not self.cap.isOpened():
        #     print("[CameraViewport] GStreamer failed, falling back to default camera")
        #     self.cap = cv2.VideoCapture(0)
        self.cap = None
        self.test_image = cv2.imread("test_image.jpg")
        if self.test_image is None:
            print("[CameraViewport] Failed to load test_image.jpg")
        else:
            print("[CameraViewport] Loaded test_image.jpg for simulation")
            
    def _update_frame(self):
        """采集帧、绘制检测框、显示"""
        # if self.cap is None or not self.cap.isOpened():
        #     return
            
        # ret, frame = self.cap.read()
        # if not ret:
        #     return

        if self.test_image is None:
            return
        frame = self.test_image.copy()
        
        # Store frame in stateBus for API uploads
        stateBus.set_frame(frame)
        
        """如果推理启用，将帧送去推理"""
        if stateBus.get_inference_enabled():
            self.inference_thread.set_frame(frame)
        
            """在帧上绘制检测框"""
            for det in self._results:
                x1, y1 = det["x1"], det["y1"]
                x2, y2 = det["x2"], det["y2"]
                class_name = det["class"]
                confidence = det["confidence"]
                
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签（在标签下面绘制填充矩形以提高可读性）
                label = f"{class_name} {confidence:.2f}"
                text_x, text_y = x1, max(20, y1 - 10)
                # 使用白色文本，避免额外计算和绘制背景矩形
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        """将 OpenCV 帧转换为 QPixmap 并显示"""
        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # 创建 QImage - 拷贝数据避免内存引用问题
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        
        # 显示到场景
        self._pix_item.setPixmap(pixmap)
        self.view.fitInView(self._pix_item, Qt.KeepAspectRatio)
        self.scene.setSceneRect(0, 0, w, h)
        
    def _on_inference_enabled_changed(self, enabled: bool):
        """处理推理启用/禁用事件"""
        if enabled:
            print("[CameraViewport] Inference enabled")
            # 只在线程未运行时启动
            if not self.inference_thread.isRunning():
                self.inference_thread.start()
        else:
            print("[CameraViewport] Inference disabled")
            self._results = []  # 清空检测结果
            
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.timer.stop()
        self.inference_thread.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
    
