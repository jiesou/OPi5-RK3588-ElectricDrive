"""Synchronous requests-based ApiClient running blocking HTTP calls in a
background thread pool. The public methods return asyncio.Future objects so
they can be awaited from Slint async callbacks. Results are delivered back to
the Slint event loop using call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
from socket import timeout
import threading
from typing import Callable, Optional, Dict, Any

import requests

from settings import stored_settings


class ApiClient:
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _base_url(self) -> str:
        base_url = stored_settings.get("api_base_url") or ""
        if not base_url:
            print("[ApiClient] API BASE URL 未配置")
            return ""
        if not base_url.endswith("/"):
            base_url += "/"
        return base_url

    def _run_async_request(self, func_exec: Callable[..., requests.Response], *args, **kwargs) -> asyncio.Future:
        """
        通用的包装器：将同步的 requests 调用丢入线程池，并将结果返回给 asyncio Future。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def worker():
            try:
                # 执行传入的 requests 调用函数
                response = func_exec(*args, **kwargs, timeout=5)
                text = response.text or ""
                
                try:
                    parsed = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    parsed = text

                if 200 <= response.status_code < 300:
                    resp = parsed if (isinstance(parsed, dict) and "success" in parsed) else {"success": True, "data": parsed}
                else:
                    error_message = parsed.get("error") if isinstance(parsed, dict) else str(parsed)
                    resp = {"success": False, "error": error_message or f"HTTP {response.status_code}"}

                loop.call_soon_threadsafe(future.set_result, resp)
            except requests.Timeout:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": "请求超时"})
            except Exception as e:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": str(e)})

        self._executor.submit(worker)
        return future

    def stop(self):
        self._executor.shutdown(wait=False)

    def upload_wiring_async(self, position: int, image_bytes: Optional[bytes] = None, result: Optional[Dict[str, int]] = None) -> asyncio.Future:
        url = self._base_url() + "cv/upload_wiring"
        
        files = {"image": ("capture.jpg", image_bytes, "image/jpeg")} if image_bytes else None
        data = {
            "position": position
        }
        # FormData 内嵌套 JSON 字符串
        if result is not None: data["result"] = json.dumps(result)

        return self._run_async_request(requests.post, url, files=files, data=data)

    def confirm_wiring_async(self) -> asyncio.Future:
        url = self._base_url() + "cv/confirm_wiring"
        return self._run_async_request(requests.post, url, json={})

    def upload_face_async(self, image: bytes, who: str) -> asyncio.Future:
        url = self._base_url() + "cv/upload_face"

        files = {"image": ("face.jpg", image, "image/jpeg")}
        data = {"who": who}

        return self._run_async_request(requests.post, url, files=files, data=data)

    def upload_deskclean_submit_async(self, image: bytes, result: Dict[str, Any]) -> asyncio.Future:
        url = self._base_url() + "cv/upload_deskclean"

        files = {"image": ("deskclean.jpg", image, "image/jpeg")}
        data = {"result": json.dumps(result)}

        return self._run_async_request(requests.post, url, files=files, data=data)

    def pull_xiaoxin_update_async(self) -> asyncio.Future:
        """拉取小新智能体状态更新"""
        url = self._base_url() + "cv/pull_xiaoxin_update"
        return self._run_async_request(requests.get, url)

api_client = ApiClient()
