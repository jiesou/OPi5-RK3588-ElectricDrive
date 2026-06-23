"""小新智能体视图，负责轮询后端状态和处理故障诊断"""

import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import slint

from api_client import api_client
from api_vl_client import vl_client
from camera_service import camera_service
from evaluation.yolo_safety import YoloSafety
# 故障类型到解决方案的映射，包含 title 和 desc
TROUBLESHOOTS: dict[str, dict[str, str]] = {
    "M1_NOT_START": {
        "title": "电动机M1不反转？",
        "desc": (
            "1. 检查电机 M1 的电源线是否正确连接。\n"
            "2. 检查变频器输出端 U/V/W 是否接对。\n"
            "3. 确认电机是否处于手动/自动模式。\n"
            "4. 检查电机是否有过载保护跳闸。"
        ),
    },
    "M1_OVERLOAD": {
        "title": "电动机M1过载？",
        "desc": (
            "1. 检查电机负载是否过大。\n"
            "2. 确认电机额定功率是否匹配。\n"
            "3. 检查机械传动部分是否卡死。"
        ),
    },
    "WIRING_ERROR": {
        "title": "接线错误？",
        "desc": (
            "1. 检查接线是否松动。\n"
            "2. 对照接线图核对每根线。\n"
            "3. 使用万用表检测通断。"
        ),
    },
}

def _invoke_on_ui_thread(callback):
    """跨线程安全调度：将 callback 调度到 Slint UI 事件循环线程执行。
    优先使用 invoke_in_main_thread，否则回退到 invoke_from_event_loop 的多个可用入口。
    """
    if hasattr(slint, "invoke_in_main_thread"):
        slint.invoke_in_main_thread(callback)
    elif hasattr(slint.native, "invoke_from_event_loop"):
        slint.native.invoke_from_event_loop(callback)
    elif hasattr(slint.slint, "invoke_from_event_loop"):
        slint.slint.invoke_from_event_loop(callback)
    else:
        callback()


class XiaoxinViewport:
    """小新智能体视图，负责轮询后端 API 并处理故障诊断流程"""

    def __init__(self):
        self._running = False
        self.latest_frame_bgr: np.ndarray | None = None
        self._last_insights_text: str = ""
        self._pull_xiaoxin_update_message_thread: Optional[threading.Thread] = None
        self._vl_thread: Optional[threading.Thread] = None
        self._safety_thread: Optional[threading.Thread] = None
        self._window = None
        self._troubleshoot_popup_closed = False
        self._alert_popup_closed = False
        self._safetycare_closed = False
        self._latest_safety_frame0: np.ndarray | None = None
        self._latest_safety_frame1: np.ndarray | None = None
        self.safety = YoloSafety()

    def start(self, window=None):
        if self._running:
            return

        self._window = window
        self._running = True
        self._pull_xiaoxin_update_message_thread = threading.Thread(target=self._pull_xiaoxin_update_message_loop, daemon=True)
        # self._pull_xiaoxin_update_message_thread.start()
        self._vl_thread = threading.Thread(target=self._vl_loop, daemon=True)
        self._vl_thread.start()
        self._safety_thread = threading.Thread(target=self._safety_loop, daemon=True)
        self._safety_thread.start()
        print("[Xiaoxin] 智能体消息更新线程启动")

    def stop(self):
        self._running = False
        if self._pull_xiaoxin_update_message_thread:
            self._pull_xiaoxin_update_message_thread.join(timeout=1.0)
        if self._vl_thread:
            self._vl_thread.join(timeout=1.0)
        if self._safety_thread:
            self._safety_thread.join(timeout=1.0)
        print("[Xiaoxin] 智能体消息更新线程停止")

    def _pull_xiaoxin_update_message_loop(self):
        """轮询后端 API 的主循环"""
        while self._running:
            # Tricks: 分拆长 sleep 为多个短 sleep，以便快速响应 stop()
            # 每 2 秒调用一次
            for _ in range(20):
                if not self._running:
                    return
                time.sleep(0.1)

            message = api_client.pull_xiaoxin_update()
            if not message:
                print("[Xiaoxin] 无法获取智能体更新消息")
                continue

            print(f"[Xiaoxin] 收到更新: type={message.type}, troubleshoot_type={message.evaluate_need_troubleshoot_type}")

            def update_ui():
                if not self._window:
                    return
                if message.type == "status_text_update":
                    self._window.XiaoxinPageData.status_text = message.status_text
                elif message.type == "evaluate_need_troubleshoot" and message.evaluate_need_troubleshoot_type:
                    if self._troubleshoot_popup_closed:
                        return
                    troubleshoot = TROUBLESHOOTS[message.evaluate_need_troubleshoot_type]
                    if not troubleshoot:
                        return
                    self._troubleshoot_popup_closed = True
                    self._window.XiaoxinPageData.troubleshoot_title = troubleshoot["title"]
                    self._window.XiaoxinPageData.troubleshoot_solution_desc = troubleshoot["desc"]
                    self._window.XiaoxinPageData.show_troubleshoot_popup = True
                elif message.type == "update_insights_text":
                    self._window.XiaoxinPageData.insights_text = message.insights_text

            _invoke_on_ui_thread(update_ui)

    SEVEN_S_LABELS = ("整理", "整顿", "清扫", "清洁", "素养", "安全", "节约")
    SEVEN_S_JSON_KEYS = ("seiri_score", "seiton_score", "seiso_score", "seiketsu_score", "shitsuke_score", "safety_score", "save_score")
    VL_PROMPT = """\
你是一个电气培训7S管理评估专家。请分析画面中学员的低压电气设备装接操作。
参照以下两个示例的推理和输出格式：先做场景观察，再生成描述，然后逐项评估7S并给出理由，
最后用 ```json 代码块输出JSON。

Example Response 1:

## 场景观察
画面中一名学员站在操作台前，正在用螺丝刀紧固配电柜内的端子排。操作台上有散落的剥线皮、几把螺丝刀和一把剥线钳，
工具没有归位。学员身穿工装但未戴安全帽，手边有一杯水。台面角落有线头堆积。

## 描述
学员站在实训板上弯腰，手拿螺丝刀，应该是在接线
但是我需要将描述控制在6个字符以内，不妨描述为“学员接线中”。

## 7S逐项评估
- 整理(Seiri)：操作台上有多余的剥线皮和线头未清理，水杯不应出现在操作台上。评分 5.5
- 整顿(Seiton)：工具散放未归位，但器件和导线摆放基本有序。评分 6.0
- 清扫(Seiso)：地面尚且干净，但台面废料堆积明显。评分 4.5
- 清洁(Seiketsu)：没有持续维护前3S的迹象，废料放任积累。评分 4.0
- 素养(Shitsuke)：未戴安全帽，工装穿戴不完全，操作习惯有待规范。评分 5.0
- 安全(Safety)：水杯在操作台上存在液体泼洒导致短路的风险；未戴安全帽。评分 3.0
- 节约(Save)：材料使用基本合理，无明显浪费。评分 9.0

接下来我需要返回精确的JSON输出：
```json
{
  "description_length": 5,
  "description": "学员接线中",
  "seiri_score": 5.5,
  "seiton_score": 6.0,
  "seiso_score": 4.5,
  "seiketsu_score": 4.0,
  "shitsuke_score": 5.0,
  "safety_score": 3.0,
  "save_score": 9.0
}
```

Example Response 2:

## 场景观察
画面中一名学员正在用万用表测量已接好线的配电柜端子导通情况。操作台上工具归位在工具架上，
剥线钳、螺丝刀各在其位。器件分类摆放在收纳盒中，导线理顺无缠绕。学员穿戴齐全：安全帽、工装、绝缘鞋均到位。
台面干净无废料，仅当前使用的万用表和图纸在台面上。

## 描述
学员站在实训板上弯腰，手拿万用表红黑表笔，应该是在测量
但是我需要将描述控制在6个字符以内，不妨描述为“测量通断中”。

## 7S逐项评估
- 整理(Seiri)：台面上仅保留当前操作必需的万用表和图纸，其余物品均收纳归位。评分 9.0
- 整顿(Seiton)：工具架上每件工具定位明确，器件按类别收纳，导线理顺无缠绕。评分 9.5
- 清扫(Seiso)：台面和地面干净整洁，无剥线皮、线头等废弃物。评分 9.0
- 清洁(Seiketsu)：前3S成果保持良好，工作区域无明显污染源。评分 8.5
- 素养(Shitsuke)：安全帽、工装、绝缘鞋穿戴齐全，操作姿势规范。评分 9.0
- 安全(Safety)：万用表使用正确，无带电裸露触点，防护到位。评分 9.9
- 节约(Save)：测量完毕后导线无明显浪费，工作节奏合理。评分 8.0

接下来我需要返回精确的JSON输出：
```json
{
  "description_length": 5,
  "description": "学员测量中",
  "seiri_score": 9.0,
  "seiton_score": 9.5,
  "seiso_score": 9.0,
  "seiketsu_score": 8.5,
  "shitsuke_score": 9.0,
  "safety_score": 9.9,
  "save_score": 8.0
}
```

现在，请对所提供的画面，按照以下步骤逐步分析：
1. 逐步推理流程：场景观察 → 描述 → 7S逐项评估 → JSON输出
2. 响应的关键是JSON输出，如果没有JSON则会产生系统崩溃
3. JSON放在 ```json 和 ``` 之间，7个评分字段缺一不可，7个分数必须做出明显区别，不能全都差不多
4. description为中文，不超过6个字，否则产生系统崩溃"""

    @staticmethod
    def _score_to_value(score: float) -> float:
        """Map 0-10 score to radar chart value (0.3-1.0)"""
        return 0.3 + max(0.0, min(10.0, score)) * 0.07

    @staticmethod
    def _parse_vl_json(text: str) -> Optional[dict]:
        """Extract JSON from VL model output containing ```json code blocks."""
        if not text:
            return None
        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[Xiaoxin] VL JSON 解析失败: {text[:200]}")
            return None

    def _apply_7s_scores(self, scores: dict, description: str) -> None:
        """Apply 7S scores from parsed JSON to radar chart data model."""
        # Build plain Python data in VL thread, construct slint.ListModel on UI thread
        axis_data: list[dict[str, object]] = []
        for label, key in zip(self.SEVEN_S_LABELS, self.SEVEN_S_JSON_KEYS):
            score = scores.get(key, 5.0)
            value = self._score_to_value(float(score))
            axis_data.append({"label": label, "value": value})

        def update_ui():
            if self._window:
                model = slint.ListModel()
                for item in axis_data:
                    model.append(item)
                self._window.XiaoxinPageData.insight_7s_data = model
                self._window.XiaoxinPageData.insights_text = description
        _invoke_on_ui_thread(update_ui)

    def _vl_loop(self):
        """VL 模型推理线程，流式分析7S并更新雷达图和响应面板"""
        print("[Xiaoxin] VL 线程启动")
        while self._running:
            for _ in range(100):
                if not self._running:
                    return
                time.sleep(0.1)

            frame = camera_service.get_frame()
            if frame is None:
                continue

            chunks: list[str] = []

            def _start_stream():
                if self._window:
                    self._window.XiaoxinPageData.vl_is_streaming = True
                    self._window.XiaoxinPageData.vl_response_text = ""
            slint.native.invoke_from_event_loop(_start_stream)

            for chunk in vl_client.analyze_image_stream(frame, prompt=self.VL_PROMPT):
                if not self._running:
                    break
                chunks.append(chunk)

                def _update_stream():
                    if self._window:
                        text = "".join(chunks)
                        self._window.XiaoxinPageData.vl_response_text = text.split("```json")[0].rstrip()
                slint.native.invoke_from_event_loop(_update_stream)

            compact_text = "".join(chunks)
            if not compact_text:
                def _clear_stream():
                    if self._window:
                        self._window.XiaoxinPageData.vl_is_streaming = False
                slint.native.invoke_from_event_loop(_clear_stream)
                continue

            parsed = self._parse_vl_json(compact_text)
            if parsed:
                description = parsed.get("description", "我在看着哦")
                self._last_insights_text = description
                self._apply_7s_scores(parsed, description)
            else:
                print(f"[Xiaoxin] VL 响应非JSON，作为纯文本展示: {compact_text[:100]}")
                description = compact_text[:6]
                self._last_insights_text = description
                def _update_fallback():
                    if self._window:
                        self._window.XiaoxinPageData.insights_text = description
                _invoke_on_ui_thread(_update_fallback)

    def _safety_loop(self):
        """Safety 模型推理线程：batch 推理两路摄像头，绘制检测框，并监控报警"""
        print("[Xiaoxin] Safety 线程启动")
        while self._running:
            if self._safetycare_closed:
                time.sleep(0.5)
                continue

            cam0 = camera_service.get_frame(0)
            cam1 = camera_service.get_frame(1)

            if cam0 is None or cam1 is None:
                time.sleep(0.001)
                continue

            results = self.safety.detect_batch([cam0, cam1])
            res0, res1 = results[0], results[1]

            drawn0 = self._draw_safety_boxes(cam0.copy(), res0.boxes)
            drawn1 = self._draw_safety_boxes(cam1.copy(), res1.boxes)
            self._latest_safety_frame0 = drawn0
            self._latest_safety_frame1 = drawn1

            alert = ""
            if not any(b.label == "workwear" for b in res0.boxes) and \
                not any(b.label == "workwear" for b in res1.boxes):
                alert = "未穿工服"
            print(res0.boxes)
            if any(b.label == "breakerON" for b in res0.boxes):
                alert = "带电接线"

            if alert and not self._alert_popup_closed:
                def _update_alert():
                    if self._window:
                        self._window.XiaoxinPageData.alert_text = alert
                        self._window.XiaoxinPageData.show_alert_popup = True
                        self._alert_popup_closed = True
                slint.native.invoke_from_event_loop(_update_alert)

    SAFETY_COLORS = {
        "workwear": (0, 255, 0),
        "breakerON": (0, 0, 255),
        "breakerOFF": (255, 0, 0),
        "person": (255, 255, 0),
    }

    @staticmethod
    def _draw_safety_boxes(frame: np.ndarray, boxes: list) -> np.ndarray:
        for box in boxes:
            color = XiaoxinViewport.SAFETY_COLORS.get(box.label, (255, 255, 255))
            overlay = frame.copy()
            cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            # label = f"{box.label} {box.conf:.2f}"
            # cv2.putText(frame, label, (box.x1, max(20, box.y1 - 6)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

# 全局单例
xiaoxin_viewport = XiaoxinViewport()

def bind_xiaoxin(window) -> None:
    """绑定小新智能体页面到窗口"""
    xiaoxin_viewport.start(window)

    @slint.callback(global_name="XiaoxinPageData")
    def request_xiaoxin_frame() -> None:
        """请求相机帧：VL 主画面 + Safety 俯拍"""
        safety0 = xiaoxin_viewport._latest_safety_frame0
        if safety0 is not None:
            xiaoxin_viewport.latest_frame_bgr = safety0
            rgb = cv2.cvtColor(safety0, cv2.COLOR_BGR2RGB)
            arr = np.ascontiguousarray(rgb, dtype=np.uint8)
            window.XiaoxinPageData.camera_frame = slint.Image.load_from_array(arr)

        safety1 = xiaoxin_viewport._latest_safety_frame1
        if safety1 is not None:
            rgb1 = cv2.cvtColor(safety1, cv2.COLOR_BGR2RGB)
            arr1 = np.ascontiguousarray(rgb1, dtype=np.uint8)
            window.XiaoxinPageData.camera_frame_safety = slint.Image.load_from_array(arr1)

    window.XiaoxinPageData.request_xiaoxin_frame = request_xiaoxin_frame

    @slint.callback(global_name="XiaoxinPageData")
    def request_safetycare_closed() -> None:
        xiaoxin_viewport._safetycare_closed = True
        window.XiaoxinPageData.safetycare_closed = True

    window.XiaoxinPageData.request_safetycare_closed = request_safetycare_closed

    @slint.callback(global_name="XiaoxinPageData")
    def request_reset_state() -> None:
        xiaoxin_viewport._troubleshoot_popup_closed = False
        xiaoxin_viewport._alert_popup_closed = False
        xiaoxin_viewport._safetycare_closed = False
        window.XiaoxinPageData.safetycare_closed = False

    window.XiaoxinPageData.request_reset_state = request_reset_state
