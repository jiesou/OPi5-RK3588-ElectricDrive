# 学生电拖测验端侧 AI 平台 ElectricDrive

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
