# 学员电气测验端侧 AI 平台 ElectricDrive

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

## 2026-05-15 7S RadarChart 实现

### 背景
xiaoxin-page 中原 7S 数据用 ProgressIndicator 列表展示，改为用 Slint Path 元素绘制雷达图。

### 关键发现
- Slint `for`-`in` 语法**不支持**在 Path 内部使用（LineTo 等 Path 子元素不能用 for 循环生成），Issue: https://github.com/slint-ui/slint/issues/754
- Slint `for`-`in` 在 Path 外部可以正常生成多个 Path 元素（如网格层级、轴线、标签）
- Path 子元素坐标是 `float` 类型（无单位），物理坐标需用 `/ 1px` 转换为 float
- Slint 支持 `sin()`/`cos()` 三角函数，参数需带 `deg`/`rad` 单位
- **Path viewbox 是 1:1 像素映射的关键**：默认情况下 Path 会将 path 坐标的 bounding box 映射到元素尺寸，导致坐标被缩放。必须显式设置 `viewbox-x/y/width/height` 才能得到 1:1 像素映射
- **Slint 不支持 `float * percent`**：`v * 1%` 会报错 "Cannot convert float to percent"，所以数据模型改用 `float` 存储比值（1.0 = 100%）
- **Slider 在 Slint 1.15.1 std-widgets 中可用**，有 `minimum`/`maximum`/`value`/`changed(float)` 属性

### 实现方案
- `radar-chart.slint`：独立 RadarChart 组件，用 Path 绘制雷达图
  - 4 级同心网格多边形（25%/50%/75%/100%）由外层 `for` 生成独立 Path
  - 数据填充多边形：硬编码 8 个 LineTo（支持最多 8 个维度），用三元条件跳过超出 axes.length 的顶点
  - 标签用 `for` 生成独立 Text 元素
  - **每个 Path 都显式设置了 viewbox 以确保 1:1 像素映射**
  - `offset: 90deg` 让第一个轴从顶部（12点钟方向）开始
- 受限于 Slint 不支持父元素遍历子元素，NavBar/NavItem 模式无法实现。采用基于 `[RadarAxis]` 数据数组的 API
- `RadarAxis` 字段：`label: string, value: float`（比值，1.0 = 100%）
- xiaoxin-page 中集成了 7 个 Slider 拖动条，可以直接拖动修改 7S 数据并实时观察雷达图变化
- Slint 版本：1.15.1b1
