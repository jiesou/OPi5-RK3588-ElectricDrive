"""Vision-Language API Client for AI image description.
Calls Qwen3-VL-Flash or similar models to describe camera frames.
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests

from settings import stored_settings

IMG_SIZE = (640, 640)


class VLClient:
    """Vision-Language model client for image description."""

    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()

    def _get_config(self) -> tuple[str, str, str]:
        """Get VL config: (api_key, base_url, model)"""
        api_key = stored_settings.get("vl_api_key") or ""
        base_url = stored_settings.get("vl_base_url") or ""
        model = stored_settings.get("vl_model") or "qwen-vl-plus"

        if not api_key or not base_url:
            print("[VLClient] VL API 未配置 (需要 vl_api_key, vl_base_url, vl_model)")
            return "", "", ""

        # 确保 base_url 不以 / 结尾
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        return api_key, base_url, model

    def analyze_image(self, frame: np.ndarray, prompt: str = "请简要描述这个场景中发生了什么") -> Optional[str]:
        """Synchronously describe an image using VL model.

        Args:
            frame: BGR image as numpy array (OpenCV format)
            prompt: Question or prompt for the model

        Returns:
            Description text or None on error
        """
        api_key, base_url, model = self._get_config()
        if not api_key:
            return None

        t_pre_start = time.perf_counter()

        # 缩放到 640x640
        frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # 编码为 JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            print("[VLClient] 图片编码失败")
            return None

        encoded_bytes = buffer.tobytes()
        image_base64 = base64.b64encode(encoded_bytes).decode("utf-8")
        t_pre_end = time.perf_counter()

        # 构建请求体（OpenAI 兼容格式）
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        t_inf_start = time.perf_counter()
        try:
            response = requests.post(
                base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            t_inf_end = time.perf_counter()

            if response.status_code == 200:
                result = response.json()
                # OpenAI 兼容响应格式
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    total_ms = (t_inf_end - t_pre_start) * 1000.0
                    print(f"[VLClient] Timing (ms): preprocess={(t_pre_end - t_pre_start) * 1000.0:.2f} inference={(t_inf_end - t_inf_start) * 1000.0:.2f} total={total_ms:.2f}")
                    print(f"[VLClient] 模型描述: {content}")
                    return content
            else:
                print(f"[VLClient] API 错误: {response.status_code} - {response.text[:200]}")

        except requests.Timeout:
            print("[VLClient] 请求超时")
        except Exception as e:
            print(f"[VLClient] 请求失败: {e}")

        return None

    def analyze_image_stream(self, frame: np.ndarray, prompt: str = ""):
        """Stream image analysis results token by token.

        Uses OpenAI-compatible streaming (SSE). Yields content strings as they arrive.
        """
        api_key, base_url, model = self._get_config()
        if not api_key:
            return

        t_pre_start = time.perf_counter()

        frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            print("[VLClient] 图片编码失败")
            return

        image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        t_pre_end = time.perf_counter()

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 2048,
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                base_url,
                json=payload,
                headers=headers,
                timeout=(10, 60),
                stream=True
            )

            if response.status_code != 200:
                print(f"[VLClient] Stream API 错误: {response.status_code} - {response.text[:200]}")
                return

            t_first_token = None
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            ttft_ms = (t_first_token - t_pre_start) * 1000.0
                            print(f"[VLClient] Stream TTFT (ms): {ttft_ms:.2f}")
                        yield content
                except json.JSONDecodeError:
                    continue

            total_ms = (time.perf_counter() - t_pre_start) * 1000.0
            print(f"[VLClient] Stream total (ms): {total_ms:.2f}")

        except requests.Timeout:
            print("[VLClient] Stream 请求超时")
        except Exception as e:
            print(f"[VLClient] Stream 请求失败: {e}")

    def stop(self):
        self._executor.shutdown(wait=False)


# Global singleton
vl_client = VLClient()
