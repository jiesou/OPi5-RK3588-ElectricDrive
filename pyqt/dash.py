import sys
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from .camera_viewport import CameraViewport
from .shots_status import ShotsStatusWidget
from .settings import SettingsWidget
from .udp_frame_uploader import uploader


class MainDash(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电拖装接评估")
        self.resize(760, 540)

        # 左侧：摄像机视图
        self.camera = CameraViewport()

        # 右侧列：拍照状态和设置
        self.shots = ShotsStatusWidget()
        self.settings = SettingsWidget()

        right_col = QVBoxLayout()
        right_col.addWidget(self.settings)
        right_col.addWidget(QLabel("拍照状态"))
        right_col.addWidget(self.shots)
        right_col.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera, stretch=2)

        right_widget = QWidget()
        right_widget.setLayout(right_col)
        main_layout.addWidget(right_widget, stretch=1)

        # 顶部工具栏（右上角：全屏切换和关闭按钮）
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        # 全屏切换按钮
        self.full_btn = QPushButton("⛶")
        self.full_btn.setFixedSize(32, 28)
        self.full_btn.setToolTip("切换全屏")
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        top_bar.addWidget(self.full_btn)
        # 关闭按钮
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 28)
        close_btn.setToolTip("关闭窗口")
        close_btn.clicked.connect(self.close)
        top_bar.addWidget(close_btn)

        # 把顶栏和主内容放进垂直布局
        main_vlayout = QVBoxLayout()
        main_vlayout.addLayout(top_bar)
        main_vlayout.addLayout(main_layout)

        self.setLayout(main_vlayout)
        
    def toggle_fullscreen(self):
        """切换窗口全屏/正常显示，并更新按钮文字。"""
        try:
            if self.isFullScreen():
                self.showNormal()
                # 使用较小的符号表明非全屏
                self.full_btn.setText("⊡")
            else:
                self.showMaximized()
                self.showFullScreen()
                self.full_btn.setText("⛶")
        except Exception:
            # 保守处理：不要因为切换全屏导致整个程序崩溃
            pass
    
    def closeEvent(self, event):
        """窗口关闭时停止上传器"""
        uploader.stop()
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    # Force the style to be the same on all OSs:
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    win = MainDash()
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
