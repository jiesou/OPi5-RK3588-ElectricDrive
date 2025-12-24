"""工艺打分 Evaluation 子模块入口"""

from .camera_viewport import camera_viewport, bind_camera
from .shots_status_widget import bind_shots_status
from .settings_widget import bind_settings
from .udp_frame_uploader import uploader

__all__ = [
    "camera_viewport",
    "bind_camera",
    "bind_shots_status",
    "bind_settings",
    "uploader",
]
