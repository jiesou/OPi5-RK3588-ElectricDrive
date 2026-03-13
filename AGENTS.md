# 学生电拖测验端侧 AI 平台 ElectricDrive

一切都发生在 slintui 下，pyqt 弃用了
Python 工具链用 uv
Slint 相关文档可以通过 context7 mcp 来获取
这个代码是在 OrangePi 5 Plus 上部署的，PC 上，你可以用 python main_debug.py 来调试 slint 部分， slint 在 .venv 里
将你对存储库的理解和对问题的理解、操作的进度存放到工程根目录的 AGENTS.md 下

## 代码风格约定

### Viewport 线程模式
- 参考 `slintui/facesignin/face_signin_viewport.py` 的写法
- 后台轮询线程方法命名为 `_xxx_loop`
- 在循环内调用具体方法处理，减少 try/except 嵌套
- 使用 `requests` 同步请求，设置 `timeout=3`

### API 调用
- 异步场景使用 `api_client.xxx_async()` 方法

### 相机帧回调
- 参考 `slintui/deskclean/deskclean_viewport.py` 的 `request_xxx_frame` 写法
- 使用 `camera_service.get_frame()` 获取帧
- 使用 `cv2.cvtColor` + `np.ascontiguousarray` + `slint.Image.load_from_array` 转换

### 数据类型化

- 服务器返回的消息使用 `@dataclass` 定义，如 `CvClientXiaoxinUpdateMessage`
- 静态配置（如故障解决方案）直接写格式化好的文本，不需要运行时转换

## 2026-03-13 xiaoxin 线程崩溃修复记录

- 问题现象：`xiaoxin_viewport.py` 后台轮询线程中更新 `window.XiaoxinPageData`，触发 `ComponentInstance is unsendable` 跨线程崩溃。
- 第一版修复问题：使用了 `slint.invoke_in_main_thread`，但当前环境 `slint` 顶层无该 API，抛出 `AttributeError`。
- 文档与运行时确认：当前环境可用 `invoke_from_event_loop`，路径为 `slint.slint.invoke_from_event_loop`（`slint.native.invoke_from_event_loop` 也可用）。
- 本次修复：新增 `_invoke_on_ui_thread()` 兼容调度函数，优先尝试 `invoke_in_main_thread`，否则回退到 `invoke_from_event_loop` 的多个可用入口；轮询线程内只做数据拉取，UI 更新统一调度回 UI 线程执行。
