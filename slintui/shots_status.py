from __future__ import annotations

from dataclasses import dataclass
from typing import List

import slint

from camera_viewport import camera_viewport
from yolo import yolo


@dataclass
class Shot:
    terminal: int = 0
    cross: int = 0
    excopper: int = 0
    exterminal: int = 0


_shots: List[Shot] = []
_shots_model: slint.ListModel[str] = slint.ListModel([])


def _format_shot(i: int, s: Shot) -> str:
    return (
        f"照片 {i}: 号码管={s.terminal} 交叉={s.cross} "
        f"露铜={s.excopper} 露端={s.exterminal}"
    )


def _rebuild_model() -> None:
    # Keep the same ListModel instance (so bindings stay intact).
    del _shots_model[:]
    for idx, s in enumerate(_shots, start=1):
        _shots_model.append(_format_shot(idx, s))


def _totals() -> Shot:
    t = Shot()
    for s in _shots:
        t.terminal += int(s.terminal)
        t.cross += int(s.cross)
        t.excopper += int(s.excopper)
        t.exterminal += int(s.exterminal)
    return t


def bind_shots_status(window) -> None:
    """Bind shot status logic to the Slint window."""

    window.current_position = 1
    window.inference_enabled = False
    window.shots = _shots_model
    window.current_detection_text = "当前: 号码管=0 交叉=0 露铜=0 露端=0"
    window.totals_text = "总计: 号码管=0 交叉=0 露铜=0 露端=0"

    def set_position(pos: int) -> None:
        if pos in (1, 2, 3):
            window.current_position = pos

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
        # Get latest frame from camera module.
        frame = camera_viewport.get_latest_frame()
        if frame is None:
            return

        pos = int(window.current_position)

        if bool(window.inference_enabled):
            det = yolo.detect(frame)
            c = det.counts
        else:
            c = {"terminal": 0, "cross": 0, "excopper": 0, "exterminal": 0}

        # Mirror legacy business rules per position.
        if pos == 1:
            shot = Shot(terminal=20, cross=int(c.get("cross", 0)))
        elif pos == 2:
            shot = Shot(terminal=int(c.get("terminal", 0)))
        elif pos == 3:
            shot = Shot(terminal=18, exterminal=int(c.get("exterminal", 0)))
        else:
            shot = Shot(
                terminal=int(c.get("terminal", 0)),
                cross=int(c.get("cross", 0)),
                excopper=int(c.get("excopper", 0)),
                exterminal=int(c.get("exterminal", 0)),
            )

        _shots.append(shot)
        _shots_model.append(_format_shot(len(_shots), shot))

        t = _totals()
        window.totals_text = (
            f"总计: 号码管={t.terminal} 交叉={t.cross} 露铜={t.excopper} 露端={t.exterminal}"
        )

        # Auto-advance position.
        if pos < 3:
            window.current_position = pos + 1

    window.set_position = set_position
    window.toggle_inference = toggle_inference
    window.capture_shot = capture_shot
    window.clear_shots = clear_shots
