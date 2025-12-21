#!/usr/bin/env python3
"""独立测试 aiohttp 是否正常工作"""

import asyncio
import aiohttp

async def test_direct():
    """直接测试 aiohttp"""
    print("[Test] 开始直接 aiohttp 测试...")
    url = "http://192.168.11.192:8000/api/cv/upload_wiring"
    
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print(f"[Test] 发送 POST 请求到 {url}")
            async with session.post(url, data={}) as resp:
                print(f"[Test] 响应状态: {resp.status}")
                text = await resp.text()
                print(f"[Test] 响应内容: {text[:200]}")
    except asyncio.TimeoutError:
        print("[Test] 超时！")
    except Exception as e:
        print(f"[Test] 错误: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("测试 1: 直接 asyncio.run()")
    asyncio.run(test_direct())
    print("=" * 50)
