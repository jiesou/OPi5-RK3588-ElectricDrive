import asyncio
import json
from typing import Optional, Dict

import aiohttp

from settings import stored_settings


class ApiClient:

    async def upload_wiring_async(self, image_bytes: Optional[bytes] = None, result: Optional[Dict[str, int]] = None, position: Optional[int] = None):
        base_url = stored_settings.get("api_base_url")
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
            return {"success": False, "error": "超时！"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def confirm_wiring_async(self):
        base_url = stored_settings.get("api_base_url")
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
            return {"success": False, "error": "超时！"}
        except Exception as e:
            return {"success": False, "error": str(e)}


api_client = ApiClient()
