from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QFormLayout, QVBoxLayout
from PySide6.QtCore import QSettings, Slot


class Config:
    """Simple wrapper around QSettings for storing configuration."""

    def __init__(self):
        self._qs = QSettings("electricdrive", "gui")

    def get(self, key, default=None):
        return self._qs.value(key, default)

    def set(self, key, value):
        self._qs.setValue(key, value)

    def sync(self):
        self._qs.sync()


# global singleton
stored_config = Config()


class SettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.server_ip = QLineEdit()
        self.save_btn = QPushButton("保存")

        form = QFormLayout()
        form.addRow(QLabel("SERVER IP:"), self.server_ip)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(self.save_btn)
        self.setLayout(v)

        self.save_btn.clicked.connect(self.on_save)

        self._load()

    def _load(self):
        # Load stored server IP (fallback to previous default IP)
        self.server_ip.setText(stored_config.get("server_ip", "192.168.11.162"))

    @Slot()
    def on_save(self):
        ip = self.server_ip.text().strip()
        if not ip:
            ip = "192.168.11.162"

        # If user didn't provide a port, append default 8000
        host = ip if (":" in ip) else f"{ip}:8000"

        api_base = f"http://{host}/api/"
        image_feed = f"udp://{host}"

        stored_config.set("server_ip", ip)
        stored_config.set("api_base_url", api_base)
        stored_config.set("image_feed_udp_url", image_feed)
        stored_config.sync()
