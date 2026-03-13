"""小新智能体视图，负责轮询后端状态和处理故障诊断"""

import asyncio
import threading
import time
from typing import Optional, Dict, Any, List

import slint

from api_client import api_client


# 故障类型到解决方案的映射
TROUBLESHOOT_SOLUTIONS: Dict[str, List[str]] = {
    "M1_NOT_START": [
        "检查电机 M1 的电源线是否正确连接",
        "检查变频器输出端 U/V/W 是否接对",
        "确认电机是否处于手动/自动模式",
        "检查电机是否有过载保护跳闸",
    ],
    "M1_OVERLOAD": [
        "检查电机负载是否过大",
        "确认电机额定功率是否匹配",
        "检查机械传动部分是否卡死",
    ],
    "WIRING_ERROR": [
        "检查接线是否松动",
        "对照接线图核对每根线",
        "使用万用表检测通断",
    ],
}


class XiaoxinViewport:
    """小新智能体视图，负责轮询后端 API 并处理故障诊断流程"""

    def __init__(self):
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._window = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, window=None):
        if self._running:
            return

        self._window = window
        self._running = True

        # 尝试获取当前事件循环
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True
        )
        self._poll_thread.start()
        print("[Xiaoxin] 智能体启动")

    def stop(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
        print("[Xiaoxin] 智能体停止")

    def _poll_loop(self):
        """轮询后端 API 的主循环"""
        while self._running:
            try:
                self._poll_xiaoxin_update()
            except Exception as e:
                print(f"[Xiaoxin] 轮询异常: {e}")

            # 每 2 秒轮询一次
            time.sleep(2)

    def _poll_xiaoxin_update(self):
        """调用 API 获取最新状态 (同步版本，在线程中运行)"""
        if not self._window:
            return

        try:
            # 使用同步请求
            import requests
            from settings import stored_settings

            base_url = stored_settings.get("api_base_url") or ""
            if not base_url:
                return

            if not base_url.endswith("/"):
                base_url += "/"

            url = base_url + "cv/pull_xiaoxin_update"

            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                return

            data = response.json()
            self._handle_xiaoxin_update(data)

        except requests.Timeout:
            pass
        except requests.RequestException as e:
            print(f"[Xiaoxin] 请求失败: {e}")
        except Exception as e:
            print(f"[Xiaoxin] 处理更新失败: {e}")

    def _handle_xiaoxin_update(self, data: Dict[str, Any]):
        """处理后端返回的状态更新"""
        if not self._window:
            return

        status_type = data.get("type", "idle")
        troubleshoot_type = data.get("evaluate_need_troubleshoot_type", "")

        print(f"[Xiaoxin] 收到更新: type={status_type}, troubleshoot_type={troubleshoot_type}")

        # 更新 UI 状态
        if status_type == "evaluate_need_troubleshoot" and troubleshoot_type:
            # 有新的故障需要处理
            self._window.XiaoxinPageData.status_type = status_type
            self._window.XiaoxinPageData.troubleshoot_type = troubleshoot_type
            self._window.XiaoxinPageData.show_troubleshoot_popup = True
            self._window.XiaoxinPageData.status_text = f"发现故障: {troubleshoot_type}"
        elif status_type == "idle":
            # 恢复空闲状态
            self._window.XiaoxinPageData.status_type = "idle"
            self._window.XiaoxinPageData.status_text = "小新智能体已就绪"

    def get_solution_steps(self, troubleshoot_type: str) -> List[str]:
        """获取故障对应的解决步骤"""
        return TROUBLESHOOT_SOLUTIONS.get(troubleshoot_type, [
            "请检查设备状态",
            "参考操作手册进行排查",
            "如无法解决请联系技术支持",
        ])


# 单例
xiaoxin_viewport = XiaoxinViewport()


def bind_xiaoxin(window) -> None:
    """绑定小新智能体页面到窗口"""

    def start_xiaoxin():
        xiaoxin_viewport.start(window)

    def stop_xiaoxin():
        xiaoxin_viewport.stop()

    # 启动智能体
    xiaoxin_viewport.start(window)

    @slint.callback(global_name="XiaoxinPageData")
    def help_me_solve() -> None:
        """用户点击'帮我解决故障'按钮"""
        troubleshoot_type = window.XiaoxinPageData.troubleshoot_type
        steps = xiaoxin_viewport.get_solution_steps(troubleshoot_type)

        # 设置解决方案步骤
        window.XiaoxinPageData.solution_steps = steps
        window.XiaoxinPageData.show_troubleshoot_popup = False
        window.XiaoxinPageData.show_solution_popup = True
        window.XiaoxinPageData.status_text = "正在排查故障..."

        print(f"[Xiaoxin] 用户请求解决故障: {troubleshoot_type}")

    @slint.callback(global_name="XiaoxinPageData")
    def close_troubleshoot_popup() -> None:
        """关闭故障发现弹窗"""
        window.XiaoxinPageData.show_troubleshoot_popup = False
        window.XiaoxinPageData.status_type = "idle"
        window.XiaoxinPageData.status_text = "小新智能体已就绪"

    # 手动绑定回调到 window
    window.XiaoxinPageData.help_me_solve = help_me_solve
    window.XiaoxinPageData.close_troubleshoot_popup = close_troubleshoot_popup
