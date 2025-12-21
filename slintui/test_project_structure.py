#!/usr/bin/env python3
"""测试：模拟项目中的模块导入顺序，找出问题根源"""

import slint
import aiohttp
import asyncio

# 简单的测试 slint 文件
SLINT_CODE = """
import { Button, LineEdit, VerticalBox, GroupBox } from "std-widgets.slint";
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

import tempfile
import os

# 在模块级别创建 slint 文件和 window（模拟项目结构）
_temp_slint = tempfile.NamedTemporaryFile(mode='w', suffix='.slint', delete=False)
_temp_slint.write(SLINT_CODE)
_temp_slint.close()

loader = slint.load_file(_temp_slint.name)
main_window = loader.TestWindow()  # 模块级别创建 window

# 模拟 bind_shots_status
def bind_callbacks(window):
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

# 在模块级别绑定（模拟项目结构）
bind_callbacks(main_window)

def main():
    try:
        main_window.show()
        main_window.run()
    finally:
        pass

if __name__ == "__main__":
    print("=== 测试: 模拟项目结构 ===")
    print("窗口已显示，请点击按钮测试。")
    
    # 方式1: 直接调用 main()
    # main()
    
    # 方式2: slint.run_event_loop(main())（项目当前用法）
    slint.run_event_loop(main())
    
    os.unlink(_temp_slint.name)
