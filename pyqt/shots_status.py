from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QCheckBox, QMessageBox,
    QButtonGroup, QRadioButton
)
from PySide6.QtCore import Slot, QTimer, Qt, Signal, QObject
import cv2
from .state_bus import stateBus, Detection, Shot
from .api_client import api_client
from .udp_frame_uploader import uploader as udp_uploader


class ShotsStatusWidget(QWidget):
    # 用于跨线程通信的信号
    _upload_finished = Signal(object)
    _confirm_finished = Signal(object)
    def __init__(self, parent=None):
        super().__init__(parent)

        # labels
        self.current_label = QLabel("当前 正在图传")
        self.totals_label = QLabel("")

        # shots list
        self.shots_list = QListWidget()
        self.shots_list.setMaximumHeight(80)

        # buttons - made larger with minimum height
        self.capture_btn = QPushButton("拍照")
        self.capture_btn.setMinimumHeight(50)
        
        self.clear_btn = QPushButton("清除")
        self.clear_btn.setMinimumHeight(50)
        
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.setMinimumHeight(50)
        
        self.cv_enable_checkbox = QCheckBox("端侧 AI 视觉推理")
        self.cv_enable_checkbox.setMinimumHeight(60)
        self.udp_enable_checkbox = QCheckBox("UDP 图传")
        self.udp_enable_checkbox.setMinimumHeight(60)
        
        # 位置选择 RadioButtons
        self.position_group = QButtonGroup(self)
        self.position_radio1 = QRadioButton("第一张")
        self.position_radio2 = QRadioButton("第二张")
        self.position_radio3 = QRadioButton("第三张")
        self.position_radio1.setChecked(True)
        self.position_group.addButton(self.position_radio1, 1)
        self.position_group.addButton(self.position_radio2, 2)
        self.position_group.addButton(self.position_radio3, 3)
        
        # Layout for position selection
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("当前拍摄:"))
        position_layout.addWidget(self.position_radio1)
        position_layout.addWidget(self.position_radio2)
        position_layout.addWidget(self.position_radio3)
        position_layout.addStretch(1)
        
        # Layout for checkbox
        chk_layout = QHBoxLayout()
        chk_layout.addWidget(self.cv_enable_checkbox)
        chk_layout.addWidget(self.udp_enable_checkbox)
        chk_layout.addStretch(1)
        
        # Layout for buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.confirm_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.current_label)
        layout.addWidget(self.totals_label)
        layout.addLayout(position_layout)
        layout.addLayout(chk_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(self.shots_list)
        self.setLayout(layout)

        # connections
        self.cv_enable_checkbox.toggled.connect(self.on_cv_enable_toggled)
        self.udp_enable_checkbox.toggled.connect(self.on_udp_enable_toggled)
        self.capture_btn.clicked.connect(self.on_capture)
        self.clear_btn.clicked.connect(self.on_clear)
        self.confirm_btn.clicked.connect(self.on_confirm)
        stateBus.detections_changed.connect(self.on_detections_changed)
        stateBus.shots_changed.connect(self.on_shots_changed)
        self.position_group.idClicked.connect(self.on_position_changed)
        stateBus.current_position_changed.connect(self.on_position_state_changed)
        
        # 连接跨线程信号
        self._upload_finished.connect(self._handle_upload_response)
        self._confirm_finished.connect(self._handle_confirm_response)

        # Hold references to temporary message boxes so they are not GC'd
        self._active_messages = []

    @Slot(Detection)
    def on_detections_changed(self, det: Detection):
        self.current_label.setText(
            f"当前 号码管={det.terminal} 交叉={det.cross} 露铜={det.excopper} 露端={det.exterminal}"
        )

    @Slot(Shot)
    def on_shots_changed(self, shot: Shot):
        self.shots_list.clear()
        for shot in stateBus.get_shots():
            self.shots_list.addItem(QListWidgetItem(
                f"照片 {self.shots_list.count() + 1}: 号码管={shot.detection.terminal} 交叉={shot.detection.cross} 露铜={shot.detection.excopper} 露端={shot.detection.exterminal}"
            ))

        totals = stateBus.get_shots_totals()
        self.totals_label.setText(
            f"总计: 号码管={totals.detection.terminal} 交叉={totals.detection.cross} 露铜={totals.detection.excopper} 露端={totals.detection.exterminal}"
        )

    def on_position_changed(self, id: int):
        """用户手动切换拍摄位置"""
        stateBus.set_current_position(id)

    @Slot(int)
    def on_position_state_changed(self, position: int):
        """同步 UI 与 stateBus 的位置状态"""
        if position == 1:
            self.position_radio1.setChecked(True)
        elif position == 2:
            self.position_radio2.setChecked(True)
        elif position == 3:
            self.position_radio3.setChecked(True)

    def on_capture(self):
        """拍摄一张图片，记作 Shot，并上传到服务器"""
        # 获取当前检测结果和帧
        det = stateBus.get_detections()
        frame = stateBus.get_frame()
        position = stateBus.get_current_position()
        
        # 1. 图片编码：将 numpy 数组 (OpenCV frame) 转为 JPEG 二进制数据
        # 这是必须的一步，即使是 Base64 方案也需要先 encode
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        # 根据是否启用端侧推理，决定是否发送 result
        inference_enabled = stateBus.get_inference_enabled()

        if inference_enabled:
            # 端侧已启用推理：一并上传推理结果
            result = {
                "sleeves_num": det.terminal,
                "cross_num": det.cross,
                "excopper_num": det.excopper,
                "exterminal_num": det.exterminal
            }
            print(f"[ShotsStatus] 端侧推理启用，上传图像与推理结果到后端 (position={position})")
            future = api_client.upload_wiring_async(image_bytes=frame_bytes, result=result, position=position)
        else:
            # 端侧未启用：仅上传图像，不传 result，让后端进行推理
            print(f"[ShotsStatus] 端侧推理未启用，上传图像仅让后端推理 (position={position})")
            future = api_client.upload_wiring_async(image_bytes=frame_bytes, result=None, position=position)
        
        # 保存当前 position 用于回调处理
        self._pending_position = position
        
        # 使用回调处理异步结果，通过信号安全地调度到主线程
        future.add_done_callback(self._on_upload_done)

    def _on_upload_done(self, future):
        """上传完成回调（在后台线程执行），发射信号到主线程"""
        try:
            response = future.result()
        except Exception as e:
            response = {"success": False, "error": str(e)}
        self._upload_finished.emit(response)

    @Slot(object)
    def _handle_upload_response(self, response):
        """处理上传响应（在主线程执行）"""
        if not response.get("success"):
            error = response.get("error", "未知错误")
            self._show_temporary_message(error, title="上传错误")
            return
        
        position = getattr(self, '_pending_position', 1)
        self._show_temporary_message(f"第{position}张照片上传成功！", timeout_ms=1500, title="提示")
        print(f"[ShotsStatus] 照片上传成功 (position={position})。服务器响应: {response}")

        # 根据 position 过滤 detection 记录
        # 第一张：只记录 cross，terminal=20, excopper=0, exterminal=0
        server_data = response.get("data", {})
        
        if position == 1:
            shot = Shot(detection=Detection(
                terminal=20,
                cross=server_data.get("cross_num", 0),
                excopper=0,
                exterminal=0
            ))
        # 第二张：只记录 terminal，cross=0, excopper=0, exterminal=0
        elif position == 2:
            shot = Shot(detection=Detection(
                terminal=server_data.get("sleeves_num", 0),
                cross=0,
                excopper=0,
                exterminal=0
            ))
        # 第三张：只记录 exterminal，terminal=18, cross=0, excopper=0
        elif position == 3:
            shot = Shot(detection=Detection(
                terminal=18,
                cross=0,
                excopper=0,
                exterminal=server_data.get("exterminal_num", 0)
            ))
        else:
            # fallback: 使用原始数据
            shot = Shot(detection=Detection(
                terminal=server_data.get("sleeves_num", 0),
                cross=server_data.get("cross_num", 0),
                excopper=server_data.get("excopper_num", 0),
                exterminal=server_data.get("exterminal_num", 0)
            ))
        
        stateBus.add_shot(shot)
        
        # 自动切换到下一张（如果未到第三张）
        if position < 3:
            stateBus.set_current_position(position + 1)

    def on_clear(self):
        """清除拍照记录"""
        stateBus.clear_shots()
    
    def on_confirm(self):
        """确认装接评估，获取最终结果"""
        future = api_client.confirm_wiring_async()
        # 使用回调处理异步结果，通过信号安全地调度到主线程
        future.add_done_callback(self._on_confirm_done)

    def _on_confirm_done(self, future):
        """确认完成回调（在后台线程执行），发射信号到主线程"""
        try:
            response = future.result()
        except Exception as e:
            response = {"success": False, "error": str(e)}
        self._confirm_finished.emit(response)

    @Slot(object)
    def _handle_confirm_response(self, response):
        """处理确认响应（在主线程执行）"""
        if not response.get("success"):
            error = response.get("error", "未知错误")
            self.totals_label.setText("当前 正在图传")
            self._show_temporary_message(error, title="确认错误")
            return
        self._show_temporary_message("确认成功！", timeout_ms=1500, title="提示")
        result = response.get("data", {})
        print(f"[ShotsStatus] 评估完成: {result}")
        
        # 显示结果给用户
        scores = result.get("scores", 0)
        no_sleeves = result.get("no_sleeves_num", 0)
        cross = result.get("cross_num", 0)
        excopper = result.get("excopper_num", 0)
        exterminal = result.get("exterminal_num", 0)
        
        self.totals_label.setText(
            f"最终评估: 得分{scores}分。号码管未标{no_sleeves}处，交叉{cross}处，露铜{excopper}处，露端子{exterminal}处"
        )

    def _show_temporary_message(self, text: str, timeout_ms: int = 2000, title: str = "提示"):
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle(title)
            msg.setText(text)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setModal(False)
            # Make it look lighter (tool window) so it behaves like a temporary message
            msg.setWindowFlags(msg.windowFlags() | Qt.Tool | Qt.WindowStaysOnTopHint)
            # Ensure widget is deleted on close to avoid leaks
            msg.setAttribute(Qt.WA_DeleteOnClose)
            msg.show()

            # Use a lambda to ensure bound method remains callable and to prevent PySide edge-cases
            QTimer.singleShot(timeout_ms, lambda: msg.close())
        except Exception as e:
            # Fall back to console logging if UI fails for any reason
            print(f"[ShotsStatus] 无法显示临时消息: {e}")

    def on_cv_enable_toggled(self, checked: bool):
        # toggle inference in stateBus (CameraViewport listens to this)
        try:
            stateBus.set_inference_enabled(bool(checked))
        except Exception:
            pass

    def on_udp_enable_toggled(self, checked: bool):
        """切换 UDP 图传：启用时调用 uploader.start(), 禁用时调用 uploader.stop()"""
        try:
            if checked:
                udp_uploader.start()
                self._show_temporary_message("UDP 图传 已启用", timeout_ms=1200, title="提示")
            else:
                udp_uploader.stop()
                self._show_temporary_message("UDP 图传 已停止", timeout_ms=1200, title="提示")
        except Exception as e:
            print(f"[ShotsStatus] UDP 切换失败: {e}")
            try:
                self._show_temporary_message(f"UDP 切换失败: {e}", title="错误")
            except Exception:
                pass

