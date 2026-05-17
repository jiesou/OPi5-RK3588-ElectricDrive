"""桌面清洁子模块"""

from .deskclean_viewport import (
    deskclean_viewport,
    bind_deskclean,
)
from .yolo_tools import yolo_tools

__all__ = [
    "bind_deskclean",
    "deskclean_viewport",
    "yolo_tools",
]
