from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import slint
import cv2
import os
import datetime

from camera_service import camera_service

from .camera_viewport import camera_viewport
from .yolo import yolo, Detection
from api_client import api_client
from udp_frame_uploader import uploader


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
    window.inference_enabled = True
    window.udp_enabled = False
    window.shots = _shots_model
    window.current_detection_text = "当前: 号码管=0 交叉=0 露铜=0 露端=0"
    window.totals_text = "总计: 号码管=0 交叉=0 露铜=0 露端=0"

    @slint.callback
    def set_shot_position(pos: int) -> None:
        if pos in (1, 2, 3):
            window.current_shot_position = pos

    @slint.callback
    def toggle_inference(enabled: bool) -> None:
        window.inference_enabled = bool(enabled)

    @slint.callback
    def toggle_udp(enabled: bool) -> None:
        """切换 UDP 图传"""
        try:
            if enabled:
                uploader.start()
                window.show_temporary_message("UDP 图传已启用")
            else:
                uploader.stop()
                window.show_temporary_message("UDP 图传已停止")
        except Exception as e:
            print(f"[ShotsStatus] UDP 切换失败: {e}")
            window.show_temporary_message(f"UDP 切换失败: {e}")

    @slint.callback
    def clear_shots() -> None:
        _shots.clear()
        _rebuild_model()
        t = _totals()
        window.totals_text = (
            f"总计: 号码管={t.terminal} 交叉={t.cross} 露铜={t.excopper} 露端={t.exterminal}"
        )

    @slint.callback
    async def capture_shot() -> None:
        """拍照并上传到服务器 - 使用异步处理"""
        frame = camera_viewport.latest_frame_bgr
        if frame is None:
            print("[ShotsStatus] 无可用帧，拍照失败")
            return

        pos = int(window.current_shot_position)

        # 编码为 JPEG
        frame_bytes: bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        # 根据是否启用推理决定是否发送 result
        inference_enabled = bool(window.inference_enabled)

        if inference_enabled:
            # 端侧已启用推理：一并上传推理结果
            detection = yolo.latest_result.detection
            result = {
                "sleeves_num": detection.terminal,
                "cross_num": detection.cross,
                "excopper_num": detection.excopper,
                "exterminal_num": detection.exterminal
            }
            print(f"[ShotsStatus] 端侧推理启用，上传图像与推理结果到后端 (position={pos})")
            response = await api_client.upload_wiring_async(position=pos, image_bytes=frame_bytes, result=result)
        else:
            # 端侧未启用：仅上传图像，不传 result，让后端进行推理
            print(f"[ShotsStatus] 端侧推理未启用，上传图像仅让后端推理 (position={pos})")
            response = await api_client.upload_wiring_async(position=pos, image_bytes=frame_bytes, result=None)

        if not response.get("success"):
            error = response.get("error", "未知错误")
            print(f"[ShotsStatus] 上传错误: {error}")
            window.show_temporary_message(f"上传错误: {error}")
            return

        print(f"[ShotsStatus] 照片上传成功 (position={pos})。服务器响应: {response}")

        # 根据 position 过滤 detection 记录
        server_data = response.get("data", {})

        if pos == 1:
            shot = Shot(detection=Detection(
                terminal=20,
                cross=server_data.get("cross_num", 0),
                excopper=0,
                exterminal=0
            ))
        elif pos == 2:
            shot = Shot(detection=Detection(
                terminal=server_data.get("sleeves_num", 0),
                cross=0,
                excopper=0,
                exterminal=0
            ))
        elif pos == 3:
            shot = Shot(detection=Detection(
                terminal=18,
                cross=0,
                excopper=0,
                exterminal=server_data.get("exterminal_num", 0)
            ))
        else:
            shot = Shot(detection=Detection(
                terminal=server_data.get("sleeves_num", 0),
                cross=server_data.get("cross_num", 0),
                excopper=server_data.get("excopper_num", 0),
                exterminal=server_data.get("exterminal_num", 0)
            ))

        _shots.append(shot)
        _shots_model.append(_format_shot(len(_shots), shot))

        # 更新总计
        t = _totals()
        window.totals_text = (
            f"总计: 号码管={t.terminal} 交叉={t.cross} 露铜={t.excopper} 露端={t.exterminal}"
        )

        # 显示成功消息
        window.show_temporary_message(f"第{pos}张照片上传成功！")

        # 自动切换到下一张（如果未到第三张）
        if pos < 3:
            window.current_shot_position = pos + 1

    @slint.callback
    def capture_dataset() -> None:
        """将当前帧保存为 JPEG 到 ./dataset 目录（同步操作）"""
        frame = camera_service.get_frame()
        if frame is None:
            print("[ShotsStatus] 无可用帧，采集失败")
            window.show_temporary_message("无可用帧，采集失败")
            return

        try:
            os.makedirs("dataset", exist_ok=True)
            fname = datetime.datetime.now().strftime("dataset/%Y%m%d_%H%M%S_%f.jpg")
            # 使用 OpenCV 保存 BGR 图像为 JPEG
            ok = cv2.imwrite(fname, frame)
            if ok:
                print(f"[ShotsStatus] 已保存图片到 {fname}")
                window.show_temporary_message(f"已保存: {os.path.basename(fname)}")
            else:
                raise RuntimeError("cv2.imwrite 返回 False")
        except Exception as e:
            print(f"[ShotsStatus] 保存失败: {e}")
            window.show_temporary_message(f"保存失败: {e}")

    @slint.callback
    async def confirm_shots() -> None:
        """确认装接评估，获取最终结果 - 使用异步处理"""
        response = await api_client.confirm_wiring_async()

        if not response.get("success"):
            error = response.get("error", "未知错误")
            print(f"[ShotsStatus] 确认错误: {error}")
            window.show_temporary_message(f"确认错误: {error}")
            return

        result = response.get("data", {})
        print(f"[ShotsStatus] 评估完成: {result}")

        # 显示结果给用户
        scores = result.get("scores", 0)
        no_sleeves = result.get("no_sleeves_num", 0)
        cross = result.get("cross_num", 0)
        excopper = result.get("excopper_num", 0)
        exterminal = result.get("exterminal_num", 0)

        window.totals_text = (
            f"最终评估: 得分{scores}分。号码管未标{no_sleeves}处，"
            f"交叉{cross}处，露铜{excopper}处，露端子{exterminal}处"
        )
        window.show_temporary_message("确认成功！")

    window.set_shot_position = set_shot_position
    window.toggle_inference = toggle_inference
    window.toggle_udp = toggle_udp
    window.capture_shot = capture_shot
    window.capture_dataset = capture_dataset
    window.clear_shots = clear_shots
    window.confirm_shots = confirm_shots
