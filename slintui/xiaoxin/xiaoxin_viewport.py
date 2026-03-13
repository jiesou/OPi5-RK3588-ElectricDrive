"""小新智能体视图，负责轮询后端状态和处理故障诊断"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import slint

from api_client import api_client
from camera_service import camera_service


# 故障类型到解决方案的映射
TROUBLESHOOT_SOLUTIONS: dict[str, str] = {
    "M1_NOT_START": "1. 检查电机 M1 的电源线是否正确连接\n2. 检查变频器输出端 U/V/W 是否接对\n3. 确认电机是否处于手动/自动模式\n4. 检查电机是否有过载保护跳闸",
    "M1_OVERLOAD": "1. 检查电机负载是否过大\n2. 确认电机额定功率是否匹配\n3. 检查机械传动部分是否卡死",
    "WIRING_ERROR": "1. 检查接线是否松动\n2. 对照接线图核对每根线\n3. 使用万用表检测通断",
}


@dataclass
class CvClientXiaoxinUpdateMessage:
    """后端轮询返回的状态更新"""
    type: str = "idle"
    evaluate_need_troubleshoot_type: str = ""


class XiaoxinViewport:
    """小新智能体视图，负责轮询后端 API 并处理故障诊断流程"""

    def __init__(self):
        self._running = False
        self._pull_xiaoxin_update_message_thread: Optional[threading.Thread] = None
        self._window = None

    def start(self, window=None):
        if self._running:
            return

        self._window = window
        self._running = True
        self._pull_xiaoxin_update_message_thread = threading.Thread(target=self._pull_xiaoxin_update_message_loop, daemon=True)
        self._pull_xiaoxin_update_message_thread.start()
        print("[Xiaoxin] 智能体消息更新线程启动")

    def stop(self):
        self._running = False
        if self._pull_xiaoxin_update_message_thread:
            self._pull_xiaoxin_update_message_thread.join(timeout=1.0)
        print("[Xiaoxin] 智能体消息更新线程停止")

    def _pull_xiaoxin_update_message_loop(self):
        """轮询后端 API 的主循环"""
        while self._running:
            # Tricks: 分拆长 sleep 为多个短 sleep，以便快速响应 stop()
            for _ in range(20):
                if not self._running:
                    return
                time.sleep(0.1)

            data = api_client.pull_xiaoxin_update()
            if not data:
                continue

            update = CvClientXiaoxinUpdateMessage(
                type=data.get("type", "idle"),
                evaluate_need_troubleshoot_type=data.get("evaluate_need_troubleshoot_type", ""),
            )

            print(f"[Xiaoxin] 收到更新: type={update.type}, troubleshoot_type={update.evaluate_need_troubleshoot_type}")

            def update_ui():
                if not self._window:
                    return
                if update.type == "evaluate_need_troubleshoot" and update.evaluate_need_troubleshoot_type:
                    self._window.XiaoxinPageData.status_type = "evaluate_need_troubleshoot"
                    self._window.XiaoxinPageData.evaluate_need_troubleshoot_type = update.evaluate_need_troubleshoot_type
                    self._window.XiaoxinPageData.show_troubleshoot_popup = True
                elif update.type == "idle":
                    self._window.XiaoxinPageData.status_type = "idle"

            slint.native.invoke_from_event_loop(update_ui)

# 全局单例
xiaoxin_viewport = XiaoxinViewport()


def bind_xiaoxin(window) -> None:
    """绑定小新智能体页面到窗口"""
    xiaoxin_viewport.start(window)

    @slint.callback(global_name="XiaoxinPageData")
    def request_xiaoxin_frame() -> None:
        """请求相机帧"""
        frame = camera_service.get_frame()
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        window.XiaoxinPageData.camera_frame = slint.Image.load_from_array(arr)

    @slint.callback(global_name="XiaoxinPageData")
    def help_me_solve() -> None:
        """用户点击'帮我解决故障'按钮"""
        evaluate_need_troubleshoot_type = window.XiaoxinPageData.evaluate_need_troubleshoot_type
        desc = TROUBLESHOOT_SOLUTIONS.get(evaluate_need_troubleshoot_type, "1. 请检查设备状态\n2. 参考操作手册进行排查\n3. 如无法解决请联系技术支持")
        window.XiaoxinPageData.troubleshoot_solution_desc = desc
        print(f"[Xiaoxin] 用户请求解决故障: {evaluate_need_troubleshoot_type}")

    window.XiaoxinPageData.request_xiaoxin_frame = request_xiaoxin_frame
    window.XiaoxinPageData.help_me_solve = help_me_solve
