import json
import os
from typing import Any


DEFAULT_SERVER_IP = "192.168.11.162"


class Settings:
    """Very small JSON-backed settings stored at PWD/settings.json."""

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(os.getcwd(), "settings.json")
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._data = {}
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get_server_ip(self) -> str:
        ip = str(self.get("server_ip", DEFAULT_SERVER_IP)).strip()
        return ip or DEFAULT_SERVER_IP

    def set_server_ip(self, ip: str) -> None:
        ip = (ip or "").strip() or DEFAULT_SERVER_IP
        host = ip if (":" in ip) else f"{ip}:8000"

        self.set("server_ip", ip)
        self.set("api_base_url", f"http://{host}/api/")
        self.set("image_feed_udp_url", f"udp://{host}")
        self.sync()

    def sync(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            # If the filesystem is read-only or path invalid, ignore.
            return


stored_settings = Settings()
