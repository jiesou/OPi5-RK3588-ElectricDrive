"""Async ApiClient using aiohttp running in a background asyncio loop thread."""
import asyncio
import json
import threading
import time
from typing import Optional, Dict, Any

import aiohttp

from .settings import stored_config


class ApiClient:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        while self._loop is None:
            time.sleep(0.01)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def upload_wiring_async(self, image_bytes: Optional[bytes] = None, result: Optional[Dict[str, int]] = None, position: Optional[int] = None):
        async def _upload():
            base_url = stored_config.get("api_base_url", "http://192.168.11.162:8000/api/")
            if not base_url.endswith('/'):
                base_url += '/'
            url = base_url + "cv/upload_wiring"
            
            try:
                data = aiohttp.FormData()
                if image_bytes:
                    data.add_field('image', image_bytes, filename='capture.jpg', content_type='image/jpeg')
                if result is not None:
                    data.add_field('result', json.dumps(result))
                if position is not None:
                    data.add_field('position', str(position))
                
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, data=data) as resp:
                        text = await resp.text()
                        print(f"[ApiClient] upload status={resp.status}, body={text[:200]}")
                        
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = text
                        
                        if 200 <= resp.status < 300:
                            if isinstance(data, dict) and 'success' in data:
                                return data
                            return {"success": True, "data": data}
                        else:
                            error = data.get('error') if isinstance(data, dict) else str(data)
                            return {"success": False, "error": error or f"HTTP {resp.status}"}
            except asyncio.TimeoutError:
                return {"success": False, "error": "网卡过热！超时！"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return asyncio.run_coroutine_threadsafe(_upload(), self._loop)

    def confirm_wiring_async(self):
        async def _confirm():
            base_url = stored_config.get("api_base_url", "http://192.168.11.162:8000/api/")
            if not base_url.endswith('/'):
                base_url += '/'
            url = base_url + "cv/confirm_wiring"
            
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json={}) as resp:
                        text = await resp.text()
                        print(f"[ApiClient] confirm status={resp.status}, body={text[:200]}")
                        
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = text
                        
                        if 200 <= resp.status < 300:
                            if isinstance(data, dict) and 'success' in data:
                                return data
                            return {"success": True, "data": data}
                        else:
                            error = data.get('error') if isinstance(data, dict) else str(data)
                            return {"success": False, "error": error or f"HTTP {resp.status}"}
            except asyncio.TimeoutError:
                return {"success": False, "error": "网卡过热！超时！"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return asyncio.run_coroutine_threadsafe(_confirm(), self._loop)

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

# Global singleton
api_client = ApiClient()
