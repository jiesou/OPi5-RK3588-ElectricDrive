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

## 2026-05-15 7S 数据雷达图实现

- **需求**: 用 Slint Path 元素绘制雷达图替换原来的 ProgressIndicator 进度条
- **Slint 版本**: 1.15.1b1 (pip)
- **实现方案**: 纯 Slint UI 层实现，不在 Python 层绘制
  - 使用 `Path` 元素的 `commands` 属性（SVG 路径语法）绘制雷达图
  - 网格层：3 层同心七边形（33%/66%/100%）
  - 轴线：7 条从中心到顶点的放射线
  - 数据填充区：根据 `XiaoxinPageData.insight_7s_data` 动态计算的七边形，带动画渐变填充
  - 数据点：每个顶点的圆形标记（白色边框 + 蓝色填充）
  - 标签：每个顶点外侧的 7S 标签文字
- **关键踩坑**:
  - `Math.cos/sin` 期望 `angle` 类型而非 `float`，需使用 `rad` 后缀如 `-1.57079633rad`
  - 不能在组件内嵌套定义 `component`（需定义在顶层或直接内联）
  - 未验证 `pure function` 语法（代码库无先例，1.15.1b1 可能不完整支持），采用内联计算
  - 7 个轴的弧度角：-1.5708, -0.6732, 0.2244, 1.1220, 2.0196, 2.9172, -2.4684（从12点方向顺时针）
