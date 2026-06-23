from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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
    scores: dict = field(default_factory=dict)


_shots: List[Shot] = []


def bind_shots_status(window) -> None:
    """绑定拍摄状态逻辑到 Slint 窗口"""

    @slint.callback(global_name="EvaluationPageData")
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

    @slint.callback(global_name="EvaluationPageData")
    def toggle_tile_inference(enabled: bool) -> None:
        yolo.tile_inference_enabled = bool(enabled)
        window.EvaluationPageData.tile_inference_enabled = bool(enabled)

    @slint.callback(global_name="EvaluationPageData")
    def clear_shots() -> None:
        _shots.clear()
        window.EvaluationPageData.submitted = False

    @slint.callback(global_name="EvaluationPageData")
    async def capture_shot() -> None:
        """拍照并上传到服务器"""
        frame = camera_viewport.latest_frame_bgr
        if frame is None:
            print("[ShotsStatus] 无可用帧，拍照失败")
            return

        # 编码为 JPEG
        frame_bytes: bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        # 上传推理结果
        detection = yolo.latest_result.detection
        result = {
            "sleeves_num": detection.terminal,
            "cross_num": detection.cross,
            "excopper_num": detection.excopper,
            "exterminal_num": detection.exterminal
        }
        print("[ShotsStatus] 上传图像与推理结果到后端")
        response = await api_client.upload_wiring_async(image_bytes=frame_bytes, result=result)

        if not response.get("success"):
            error = response.get("error", "未知")
            print(f"[ShotsStatus] 上传错误: {error}")
            window.show_temporary_message(f"上传: {error}")
            return

        print(f"[ShotsStatus] 照片上传成功。服务器响应: {response}")

        # 使用服务器响应中的 detection 记录
        server_data = response.get("data", {})
        shot = Shot(
            detection=Detection(
                terminal=server_data.get("sleeves_num", 0),
                cross=server_data.get("cross_num", 0),
                excopper=server_data.get("excopper_num", 0),
                exterminal=server_data.get("exterminal_num", 0),
            ),
            scores=server_data.get("scores", {}),
        )

        # 始终只保留一个 shot
        if _shots:
            _shots[0] = shot
        else:
            _shots.append(shot)

        window.show_temporary_message("照片上传成功！")

        if len(_shots) == 1:
            await confirm_shots()

    @slint.callback(global_name="EvaluationPageData")
    def capture_dataset() -> None:
        frame = camera_service.get_frame(0)
        if frame is None:
            print("[ShotsStatus] 无可用帧，采集失败")
            window.show_temporary_message("无可用帧，采集失败")
            return

        try:
            os.makedirs("dataset", exist_ok=True)
            fname = datetime.datetime.now().strftime("dataset/%Y%m%d_%H%M%S_%f.jpg")
            ok = cv2.imwrite(fname, frame)
            if ok:
                print(f"[ShotsStatus] 已保存图片到 {fname}")
                window.show_temporary_message(f"已保存: {os.path.basename(fname)}")
            else:
                raise RuntimeError("cv2.imwrite 返回 False")
        except Exception as e:
            print(f"[ShotsStatus] 保存失败: {e}")
            window.show_temporary_message(f"保存失败: {e}")

    @slint.callback(global_name="EvaluationPageData")
    async def confirm_shots() -> None:
        """确认装接评估，获取最终结果"""
        response = await api_client.confirm_wiring_async()

        if not response.get("success"):
            error = response.get("error", "未知")
            print(f"[ShotsStatus] 确认: {error}")
            window.show_temporary_message(f"确认: {error}")
            return

        server_data = response.get("data", {})
        print(f"[ShotsStatus] 评估完成: {server_data}")

        # 服务器回传 finalResult 覆盖最新的 shot
        if _shots:
            _shots[0].detection.terminal = server_data.get("sleeves_num", _shots[0].detection.terminal)
            _shots[0].detection.cross = server_data.get("cross_num", _shots[0].detection.cross)
            _shots[0].detection.excopper = server_data.get("excopper_num", _shots[0].detection.excopper)
            _shots[0].detection.exterminal = server_data.get("exterminal_num", _shots[0].detection.exterminal)
            _shots[0].scores = server_data.get("scores", _shots[0].scores)

        window.EvaluationPageData.submitted = True
        window.show_temporary_message("确认成功！")

    window.EvaluationPageData.clear_shots = clear_shots
    window.EvaluationPageData.toggle_udp = toggle_udp
    window.EvaluationPageData.toggle_tile_inference = toggle_tile_inference
    window.EvaluationPageData.capture_shot = capture_shot
    window.EvaluationPageData.capture_dataset = capture_dataset
