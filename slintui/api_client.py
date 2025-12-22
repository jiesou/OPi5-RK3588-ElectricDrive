"""Synchronous requests-based ApiClient running blocking HTTP calls in a
background thread pool. The public methods return asyncio.Future objects so
they can be awaited from Slint async callbacks. Results are delivered back to
the Slint event loop using call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from typing import Optional, Dict, Any

import requests

from settings import stored_settings


class ApiClient:
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

    def _build_base_url(self) -> str:
        base_url = stored_settings.get("api_base_url") or ""
        if not base_url:
            print("[ApiClient] API BASE URL 未配置")
            return ""
        if not base_url.endswith("/"):
            base_url += "/"
        return base_url

    def upload_wiring_async(self, image_bytes: Optional[bytes] = None, result: Optional[Dict[str, int]] = None, position: Optional[int] = None) -> asyncio.Future:
        """Submit a multipart upload to the server in a background thread.

        Returns an asyncio.Future that will be completed on the Slint event loop.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def worker():
            base_url = self._build_base_url()
            
            url = base_url + "cv/upload_wiring"
            print(f"[ApiClient] upload_wiring -> {url}")

            files = {}
            data: Dict[str, Any] = {}
            if image_bytes:
                files["image"] = ("capture.jpg", image_bytes, "image/jpeg")
            if result is not None:
                data["result"] = json.dumps(result)
            if position is not None:
                data["position"] = str(position)

            try:
                r = requests.post(url, files=files if files else None, data=data if data else None, timeout=5)
                text = r.text or ""
                print(f"[ApiClient] upload status={r.status_code}, body={text[:200]}")

                try:
                    parsed = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    parsed = text

                if 200 <= r.status_code < 300:
                    if isinstance(parsed, dict) and "success" in parsed:
                        resp = parsed
                    else:
                        resp = {"success": True, "data": parsed}
                else:
                    error = parsed.get("error") if isinstance(parsed, dict) else str(parsed)
                    resp = {"success": False, "error": error or f"HTTP {r.status_code}"}

                loop.call_soon_threadsafe(future.set_result, resp)
            except requests.Timeout:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": "请求超时"})
            except Exception as e:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": str(e)})

        self._executor.submit(worker)
        return future

    def confirm_wiring_async(self) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def worker():
            base_url = self._build_base_url()
            
            url = base_url + "cv/confirm_wiring"
            print(f"[ApiClient] confirm_wiring -> {url}")

            try:
                r = requests.post(url, json={}, timeout=5)
                text = r.text or ""
                print(f"[ApiClient] confirm status={r.status_code}, body={text[:200]}")

                try:
                    parsed = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    parsed = text

                if 200 <= r.status_code < 300:
                    if isinstance(parsed, dict) and "success" in parsed:
                        resp = parsed
                    else:
                        resp = {"success": True, "data": parsed}
                else:
                    error = parsed.get("error") if isinstance(parsed, dict) else str(parsed)
                    resp = {"success": False, "error": error or f"HTTP {r.status_code}"}

                loop.call_soon_threadsafe(future.set_result, resp)
            except requests.Timeout:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": "请求超时"})
            except Exception as e:
                loop.call_soon_threadsafe(future.set_result, {"success": False, "error": str(e)})

        self._executor.submit(worker)
        return future

    def stop(self):
        with self._lock:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass


api_client = ApiClient()
