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

## 2026-05-17 xiaoxin 雷达图切换页面消失修复

### 问题
来回切换界面时 xiaoxin page 的雷达图间歇性消失。

### 根因
Tab 切换时 Slint 的 `if` 块会销毁/重建 `XiaoxinPage` 及其子组件 `RadarChart`。重建后首次布局前，`self.width` 和 `self.height` 瞬态为 0，导致：
- `r` 计算为 `-72`（负半径 → 退化的多边形几何）
- `viewbox-width` / `viewbox-height` 为 `0`，Path 元素拒绝渲染或缓存了无效几何

Slint 1.15.1b1 在 Path 元素上可能不会在布局完成后正确地重新计算/重绘缓存的几何。

### 修复
- `radar-chart.slint:16`: `r` 加 `max(..., 0)` 守卫，防止负半径
- `radar-chart.slint:29-30, 76-77`: `viewbox-width/height` 加 `max(..., 1)` 守卫，防止零尺寸视口
- `xiaoxin-page.slint:374-377`: 去除雷达图容器的 `in-out` 透明度动画（出现时不再有 300ms 动画），确保页面切换回来时立即可见

### 验证
`slint.load_file("ui/app-window.slint")` 编译成功。

## 2026-05-15 xiaoxin-page VL 流式响应与摄像头布局优化

### 背景
- 摄像头画面 1440x720 (2:1)，放在右侧全高区域时 `image-fit: contain` 造成上下巨大黑边
- VL 模型返回的详细分析结果（场景观察、7S逐项评估、思维链）只提取了 JSON 评分，其余内容丢弃
- VL API 为同步阻塞请求，无流式响应，等待时间长且体验差

### 变更

#### `api_vl_client.py` — 新增流式 API
- 新增 `analyze_image_stream()` 生成器方法
- 使用 `stream: true` 开启 OpenAI 兼容流式响应（SSE）
- `max_tokens: 1024`（原 256），确保长推理不被截断
- 逐 token yield，支持首 token 延迟（TTFT）计时

#### `xiaoxin-page-data.slint` — 新增响应数据字段
- `vl-response-text: string` — 流式 VL 模型完整响应文本
- `vl-is-streaming: bool` — 流式生成状态指示

#### `xiaoxin_viewport.py` — `_vl_loop` 改用流式
- `_vl_loop` 改为调用 `analyze_image_stream()` 逐 chunk 接收
- 每个 chunk 通过 `invoke_from_event_loop` 实时更新 `vl_response_text`
- 流开始时设 `vl_is_streaming = true`，结束时设为 `false`
- 流结束后再对整个响应做 JSON 解析（提取 7S 评分），不影响原始响应展示

#### `xiaoxin-page.slint` — 右侧面板布局重构
- 右侧面板改为 VerticalLayout 上下分栏：
  - **VL 模型分析面板**（上方）：
    - 绿色/灰色圆点指示流式状态
    - `clip: true` 矩形容纳多行滚动文本
    - 空白时显示占位文本"等待 VL 模型分析…"
  - **摄像头画面**（下方）：
    - `height: parent.width / 2` 精确匹配 2:1 宽高比
    - 消除上下黑边，画面紧凑填充
- 添加 opacity 动画与左侧面板一致的展开/收起效果

### 关键技术点
- `parent.width / 2` 在 Slint 中是合法的 length 表达式，可直接用于子元素高度
- Slint `Text` 的 `vertical-alignment` 接受 `top`/`center`/`bottom`，不接受 `start`（那是 layout item 的 alignment）
- `invoke_from_event_loop` 在循环中多次调用是安全的，调用顺序与调度顺序一致

## 2026-05-17 yolo_tools 模型加载修复

### 问题
`deskclean_viewport.py` 中 `from .yolo_tools import yolo_tools, ToolBox` 报 `ModuleNotFoundError`，且 YOLO 推理无效果。

### 根因
1. **文件名含连字符**: 源文件命名为 `yolo-tools.py`，Python 无法将连字符文件名作为模块导入（`import yolo_tools` 无法匹配 `yolo-tools.py`）
2. **模型路径依赖 CWD**: `YoloTools.__init__` 使用 `"./batch1-electricdrive-tools-v10.rknn"` 相对路径，当 CWD 不是 `slintui/` 时加载失败，`self.rknn = None`，`detect()` 静默返回空结果

### 修复
- 重命名 `yolo-tools.py` → `yolo_tools.py`
- 模型路径改为基于模块文件的绝对路径: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 定位到 `slintui/` 目录

## 2026-05-19 face_signin_viewport 重构：分辨率缩放坐标错位修复

### 问题
人脸签到页面的可视化框框和实际人脸位置因为分辨率缩放而错位，代码存在多个 patch，结构混乱。

### 根因
1. **竞态条件导致缩放错位**: 推理线程独立更新 `img_scale` 和 `latest_result.faces`（无锁保护），`request_signin_frame` 回调分别读取两个值。若两次读取之间 `img_scale` 发生变化（分辨率重初始化），则 face 坐标用错误的 scale 映射到原始帧，导致框框偏移。
2. **分辨率变化检测逻辑错误**: `max(h, w) != max(self.img_size[1], self.img_size[0])` 比较的是原始帧长边与缩放后长边，这两个值永远不等，导致 `_reinit_detector` 在每个循环都被调用。
3. **`cam_id` 参数无用**: `camera_service.get_frame(cam_id)` 实现中完全忽略 `cam_id` 参数，直接返回 `self._frame`。
4. **职责混杂**: 坐标映射、绘制逻辑散落在 callback 函数中，未遵循 deskclean_viewport 的静态绘制方法模式。

### 修复
- **坐标在检测时立即映射**: 推理线程完成检测后，立即将 face 坐标从缩放空间映射到原始帧空间（除以 `_img_scale`），存入 `latest_result`。绘制时直接使用坐标，消除 `img_scale` 跨线程竞态。
- **锁保护共享结果**: 添加 `_result_lock`，使用 `get_latest_result()` 访问器，遵循 `deskclean_viewport.py` 模式。
- **修正分辨率检测**: 跟踪 `_last_cam_w` / `_last_cam_h` 记录上次检测时的原始分辨率，仅在变化时调用 `_reinit_detector`。
- **分辨率自适应绘制**: `_draw_faces` 静态方法通过 `result.frame_w` / `result.frame_h` 与当前叠加帧尺寸计算 `sx` / `sy` 缩放因子，处理检测帧与绘制帧分辨率不一致的边界情况。
- **`_draw_faces` 静态方法**: 绘制逻辑从 callback 提取到视图类，与 deskclean 架构一致。
- 去除无效的 `cam_id` 参数传递。

## 2026-06-22 Safety 模型集成与多摄像头支持

### 背景
新增 `batch2-electricdrive-safety-v10.3.rknn` 模型，在 xiaoxin 页面**同时**使用两个摄像头做安全检测。

### 变更文件

#### 新建 `slintui/evaluation/yolo_safety.py`
- `YoloSafety` 类，3 类别：`workwear` / `breakerON` / `breakerOFF`
- DFL 后处理与 `yolo.py` 相同，但**不做切图** — `pre_process` 全图 letterbox 到 640x640
- 单张推理，输出 `SafetyResult(boxes=[SafetyBox])`

#### 修改 `slintui/camera_service.py`
- **多摄像头同时抓取**：`_caps` / `_frames` / `_threads` 改为 dict，每路摄像头独立 capture 线程
- `get_frame(cam_id: int | None = None)`：`None` 返回当前选中摄像头（`_current_idx`），显式传 ID 返回指定摄像头
- `set_camera(idx)` 只更新 `_current_idx`（不再切换物理摄像头）
- 启动时自动为所有可用摄像头创建抓取线程

#### 修改 `slintui/xiaoxin/xiaoxin_viewport.py`
- 导入 `YoloSafety`，新增 `self.safety` 实例
- 新增 `_safety_loop` 线程（每 500ms）：
  - `get_frame(0)` 检测 `workwear` → 缺失则报警 "未穿工服警告！"
  - `get_frame(1)` 检测 `breakerON` → 出现则报警 "带电接线警告！"
- 报警覆盖 `status_text`；安全时恢复 `_last_status_text`
- `_last_status_text` 由后端轮询更新

### 摄像头使用说明
- 所有摄像头**常开**（不再单一切换）：idx=0 前置/人脸，idx=1 俯拍
- 向后兼容：`get_frame()` 不带参数返回 `_current_idx` 的帧（各 tab 通过 `set_camera` / `activate_tab` 切换当前索引）
- 其他 viewport 无需修改即可继续使用 `get_frame()`

## 2026-05-22 evaluation yolo.py 切图策略适配 1920x1080

### 背景
idx=0 摄像头从 1280x720 换为 1920x1080，原切图策略（缩放到 640x640 后左右切图 + 全局缩略）需要随分辨率变更。

### 变更
- `EXPECTED_H`/`EXPECTED_W`: 720/1280 → 1080/1920
- 切图策略改为左下角裁取方案：
  1. 从 1920x1080 原图左下角裁取 640x640: `bgr[h-640:h, 0:640]`
  2. 上半 320x640 → 上下各 pad 160 到 640x640（保长宽比）
  3. 下半 320x640 → 上下各 pad 160 到 640x640
  4. 全局 640x640（不 pad）
  5. 三图并行推理（source 0/1/2）
- 坐标映射 metas 同步调整：`pad_y` 加入内部 160px padding，`offset_y` 根据 crop 在 letterboxed 图中的位置计算
- `ratio_crop` 恒为 1.0（不再 resize 切图），`ratio = scale_fit` 直接控制
