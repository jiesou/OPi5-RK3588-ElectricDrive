from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import slint

from camera_viewport import camera_viewport
from yolo import yolo, Detection


@dataclass
class Shot:
    """单次拍摄记录"""
    detection: Detection = field(default_factory=Detection)


_shots: List[Shot] = []
_shots_model: slint.ListModel[str] = slint.ListModel([])


def _format_shot(i: int, s: Shot) -> str:
    d = s.detection
    return (
        f"照片 {i}: 号码管={d.terminal} 交叉={d.cross} "
        f"露铜={d.excopper} 露端={d.exterminal}"
    )


def _rebuild_model() -> None:
    """重建显示模型"""
    del _shots_model[:]
    for idx, s in enumerate(_shots, start=1):
        _shots_model.append(_format_shot(idx, s))


def _totals() -> Detection:
    """计算所有拍摄的总计"""
    total = Detection()
    for s in _shots:
        total.terminal += s.detection.terminal
        total.cross += s.detection.cross
        total.excopper += s.detection.excopper
        total.exterminal += s.detection.exterminal
    return total


def bind_shots_status(window) -> None:
    """绑定拍摄状态逻辑到 Slint 窗口"""

    window.current_shot_position = 1
    window.inference_enabled = False
    window.shots = _shots_model
    window.current_detection_text = "当前: 号码管=0 交叉=0 露铜=0 露端=0"
    window.totals_text = "总计: 号码管=0 交叉=0 露铜=0 露端=0"

    def set_shot_position(pos: int) -> None:
        if pos in (1, 2, 3):
            window.current_shot_position = pos

    def toggle_inference(enabled: bool) -> None:
        window.inference_enabled = bool(enabled)

    def clear_shots() -> None:
        _shots.clear()
        _rebuild_model()
        t = _totals()
        window.totals_text = (
            f"总计: 号码管={t.terminal} 交叉={t.cross} 露铜={t.excopper} 露端={t.exterminal}"
        )

    def capture_shot() -> None:
        frame = camera_viewport.get_latest_frame()
        if frame is None:
            return

        pos = int(window.current_shot_position)

        # 根据拍摄位置和是否启用推理来决定检测结果
        if bool(window.inference_enabled):
            result = yolo.detect(frame)
            detected = result.detection
        else:
            detected = Detection()

        # 每张图片固定不同识别结果的业务逻辑
        if pos == 1:
            # 第1个位置：固定20个号码管，检测交叉
            shot = Shot(detection=Detection(
                terminal=20,
                cross=detected.cross
            ))
        elif pos == 2:
            # 第2个位置：检测号码管
            shot = Shot(detection=Detection(
                terminal=detected.terminal
            ))
        elif pos == 3:
            # 第3个位置：固定18个号码管，检测露端
            shot = Shot(detection=Detection(
                terminal=18,
                exterminal=detected.exterminal
            ))
        else:
            # 其他位置：使用完整的检测结果
            shot = Shot(detection=detected)

        _shots.append(shot)
        _shots_model.append(_format_shot(len(_shots), shot))

        # 更新总计
        t = _totals()
        window.totals_text = (
            f"总计: 号码管={t.terminal} 交叉={t.cross} 露铜={t.excopper} 露端={t.exterminal}"
        )

        # 自动切换到下一个拍摄位置
        if pos < 3:
            window.current_shot_position = pos + 1

    window.set_shot_position = set_shot_position
    window.toggle_inference = toggle_inference
    window.capture_shot = capture_shot
    window.clear_shots = clear_shots
