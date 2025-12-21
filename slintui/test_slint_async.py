#!/usr/bin/env python3
"""测试 Slint 中 async callback 的行为"""

import slint
import aiohttp
import asyncio

# 简单的测试 slint 文件
SLINT_CODE = """
import { Button } from "std-widgets.slint";
export component TestWindow inherits Window {
    width: 300px;
    height: 200px;
    
    callback do_request();
    
    VerticalLayout {
        Button {
            text: "Test Request";
            clicked => { do_request(); }
        }
    }
}
"""

def test_with_run():
    """使用 window.run() 方式"""
    print("\n=== 测试: 使用 window.run() ===")
    
    # 动态编译 slint
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slint', delete=False) as f:
        f.write(SLINT_CODE)
        slint_file = f.name
    
    try:
        # 加载
        loader = slint.load_file(slint_file)
        window = loader.TestWindow()
        
        @slint.callback
        async def do_request():
            print("[Callback] 开始请求...")
            url = "http://192.168.11.192:8000/api/cv/upload_wiring"
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    print(f"[Callback] 发送 POST 到 {url}")
                    async with session.post(url, data={}) as resp:
                        print(f"[Callback] 响应状态: {resp.status}")
                        text = await resp.text()
                        print(f"[Callback] 响应: {text[:100]}")
            except asyncio.TimeoutError:
                print("[Callback] 超时!")
            except Exception as e:
                print(f"[Callback] 错误: {e}")
        
        window.do_request = do_request
        
        print("窗口已显示，请点击按钮测试。按 Ctrl+C 退出。")
        window.show()
        window.run()
        
    finally:
        os.unlink(slint_file)


def test_with_run_event_loop():
    """使用 slint.run_event_loop() 方式"""
    print("\n=== 测试: 使用 slint.run_event_loop() ===")
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slint', delete=False) as f:
        f.write(SLINT_CODE)
        slint_file = f.name
    
    try:
        loader = slint.load_file(slint_file)
        window = loader.TestWindow()
        
        @slint.callback
        async def do_request():
            print("[Callback] 开始请求...")
            url = "http://192.168.11.192:8000/api/cv/upload_wiring"
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    print(f"[Callback] 发送 POST 到 {url}")
                    async with session.post(url, data={}) as resp:
                        print(f"[Callback] 响应状态: {resp.status}")
                        text = await resp.text()
                        print(f"[Callback] 响应: {text[:100]}")
            except asyncio.TimeoutError:
                print("[Callback] 超时!")
            except Exception as e:
                print(f"[Callback] 错误: {e}")
        
        window.do_request = do_request
        
        print("窗口已显示，请点击按钮测试。按 Ctrl+C 退出。")
        window.show()
        slint.run_event_loop()
        
    finally:
        os.unlink(slint_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        test_with_run_event_loop()
    else:
        test_with_run()
