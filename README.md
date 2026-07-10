# 🚀 YOLO11/YOLOv8x Go 目标检测器（支持中文标签）

一个基于 **ONNX Runtime** 和 **YOLO11/YOLOv8x** 的轻量级目标检测工具，使用 Go 语言编写，支持中文标签显示、多平台（Windows/macOS/Linux）。

基于48个标准化测试程序（24 Go + 18 Python + 2 消融 + 1 72h + 1 C API + 2 Arena消融）的严格测试，累计逾80万次推理实验，本项目在同线程配置下Go与Python推理延迟几乎相同（差异<1%），Go在单实例内存上约有17%的优势。

![示例图](assets/bus_11x_false.jpg) 

## ✨ 特性

- 🖼️ 支持 JPG/PNG/GIF/BMP 输入
- 💡 自动识别中文字体，显示中文标签
- ⚡ 高性能推理（ONNX Runtime + GPU 可选）
- 🎨 彩色边界框 + 置信度标签 + 鲜明分类色彩
- 📦 跨平台（Windows / macOS / Linux）
- 🔄 多线程并发处理
- 📝 系统文本标注功能
- 📊 支持批量处理图像
- 🔧 可调节的检测参数（置信度、IOU阈值等）
- 🏎️ 检测器池机制，复用模型会话
- 📈 内存池优化，提高内存使用效率
- 🔍 支持矩形缩放和测试时增强(TTA)
- 📁 支持目录和文本文件批量输入

## 🛠️ 快速开始

### 1. 安装 Go（≥1.20）
[https://go.dev/dl](https://go.dev/dl)

### 2. 环境准备
确保系统中安装了 Go 并配置好 GOPATH 环境变量。

### 3. 克隆项目
```bash
git clone https://github.com/yourusername/yolo-go-detector.git
cd yolo-go-detector
```

### 4. 安装依赖
```bash
go mod tidy
```

### 5. 模型文件

项目支持同时使用 **YOLO11x** 和 **YOLOv8x** 模型，无需修改代码即可切换使用。

#### 支持的模型文件
- `yolo11x.onnx` - YOLO11x 模型（默认使用）
- `yolov8x.onnx` - YOLOv8x 模型

#### 手动添加模型文件
请将模型文件放置到 `./third_party/` 目录下。

**导出参数建议**:
```bash
# YOLO11x
yolo export model=yolo11x.pt format=onnx imgsz=640 opset=17

# YOLOv8x
yolo export model=yolov8x.pt format=onnx imgsz=640 opset=17
```

**注意**：默认参数下请使用 `rect=false`，本程序的 `rect=true` 仅在导出参数 `dynamic=True` 时有意义。

### 6. 编译运行
```bash
go run .
```

## ⚙️ 使用参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `-img` | `./assets/bus.jpg` | 输入图像路径、目录或.txt文件 |
| `-output` | `./assets/bus_11x_false.jpg` | 输出图像路径 |
| `-conf` | `0.25` | 置信度阈值，过滤低置信度检测结果 |
| `-iou` | `0.7` | IOU阈值，用于非极大值抑制(NMS) |
| `-size` | `640` | 模型输入尺寸，通常为640x640 |
| `-rect` | `false` | 是否使用矩形缩放（保持长宽比） |
| `-augment` | `false` | 是否启用测试时增强(TTA) |
| `-batch` | `1` | 推理的批处理大小 |
| `-workers` | `CPU核数/2` | 并发工作协程数量 |
| `-queue-size` | `100` | 任务队列大小 |
| `-timeout` | `30s` | 单个任务超时时间 |
| `-enable-system-text` | `true` | 是否显示系统文本 |
| `-system-text` | `重要设施危险场景监测系统` | 系统显示文本 |
| `-text-location` | `bottom-left` | 系统文本位置 (top-left, bottom-left, top-right, bottom-right) |

### 示例命令

检测单个图像：
```bash
go run . -img ./assets/bus.jpg -output ./output/bus_11x_false.jpg -conf 0.5
```

批量处理目录中的图像：
```bash
go run . -img ./test_images/ -conf 0.3 -workers 4
```

启用系统文本标注：
```bash
go run . -img ./assets/bus.jpg -output ./output/bus_11x_true.jpg -enable-system-text=true -system-text="智能安全监控系统" -text-location="top-left"
```

## 🏗️ 项目架构

### 核心组件

1. **主程序 (main.go)**
   - 命令行参数解析
   - 图像处理流程
   - 模型推理
   - 结果可视化

2. **检测器池 (detector_pool.go)**
   - 模型会话池管理
   - 并发任务处理
   - 工作协程管理

3. **关键功能模块**
   - 图像预处理（缩放、填充）
   - ONNX Runtime 集成
   - 非极大值抑制 (NMS)
   - 内存池优化
   - 中文标签支持

### 项目结构

```
yolo-go-detector/
├── main.go           # 主程序入口，包含检测逻辑
├── detector_pool.go  # 检测器池，支持并发处理
├── README.md         # 项目说明
├── LICENSE           # 许可证
├── .gitignore        # Git忽略文件
├── .gitattributes   # Git属性文件
├── assets/           # 资源文件（测试图像）
│   ├── bus.jpg           # 测试图像
│   ├── bus_11x_false.jpg # YOLO11x检测结果（rect=false）
│   └── bus_11x_true.jpg  # YOLO11x检测结果（rect=true）
├── results/          # 测试结果存储
│   ├── charts/                               # 图表文件（PNG和PDF格式）
│   ├── go_baseline_result.txt                # Go基准测试结果
│   ├── python_baseline_result.txt            # Python基准测试结果
│   └── ...
├── engine/           # GoYOLO-Engine核心引擎
│   ├── session_pool.go    # Session池管理 + BatchInferenceEngine
│   ├── postprocess.go     # YOLO输出解析、NMS后处理
│   ├── optimizer.go       # 性能统计组件
│   ├── tensor_pool.go     # Tensor内存池
│   └── ...
├── examples/         # engine包API使用示例
│   ├── example.go        # BatchInferenceEngine 任务提交/回调演示
│   └── README.md
├── test/             # 测试脚本和数据
│   ├── benchmark/    # Go基准测试
│   ├── charts/       # 图表生成脚本
│   ├── python/       # Python相关测试
│   ├── compare/      # Python vs Go 检测一致性对比
│   ├── batch_verify/ # 批量图片检测验证
│   └── ...
├── third_party/      # 第三方依赖
│   ├── onnxruntime.dll  # ONNX Runtime库
│   ├── yolo11x.onnx     # YOLO11x模型
│   └── yolov8x.onnx     # YOLOv8x模型
├── go.mod            # Go模块文件
└── go.sum            # Go依赖校验文件
```

## 📖 研究成果导航

本仓库是论文《面向工业监控的 ONNX Runtime 高并发推理架构设计与实现》的配套代码。以下是从论文到代码的对应关系：

### 核心架构实现

| 论文章节 | 代码位置 | 说明 |
|----------|----------|------|
| §2.3 Session Pool 并发推理架构 | `test/benchmark/go_architecture_benchmark.go` | 三种并发架构（Unsafe Shared / Mutex / Session Pool）的完整实验实现 |
| §2.3.3 Session Pool 设计 | `detector_pool.go` | 工程参考实现：`ModelSessionPool` + `VideoDetectorManager` |
| §2.3.3 会话池核心逻辑 | `engine/session_pool.go` | 架构参考实现：`SessionPool` + `BatchInferenceEngine` |
| 实际运行逻辑 | `test/benchmark/go_*.go`<br>`test/benchmark/python_*.py` | 48 个标准化测试程序（24 Go + 18 Python + 2 消融 + 1 72h + 1 C API + 2 Arena消融）各自独立实现推理，论文数据均由此产生 |

### 实验数据来源

| 论文图表 | 原始数据文件 |
|----------|-------------|
| 表1 三种架构对比 | `results/go_architecture_comparison.txt` |
| 表2 跨语言基准对比 | `results/go_baseline_result.txt`, `results/python_baseline_result.txt` |
| 图1-图7 | `results/charts/` （PNG + PDF） |
| 全部实验原始日志 | `results/*.txt` |

### 可运行入口

- **单图检测**：`go run . -img ./assets/bus.jpg`
- **批量检测**：`go run . -img ./test_images/ -workers 4`
- **批量验证（性能基准）**：`cd test/batch_verify && go build -o batch_verify.exe . && batch_verify.exe -dir <图片目录> -limit 100 -model <模型路径>`
- **全部实验复现**：`cd test && run_all_tests_complete.bat`
- **跨语言检测一致性验证**：`cd test/compare && python compare.py`（详见 [test/compare/README.md](test/compare/README.md)）

### 检测结果一致性验证

为保证 Go ONNX Runtime 推理管线的正确性，本项目完成了 **Python（Ultralytics）↔ Go（ONNX Runtime 原生 API）端到端检测一致性验证**。

**测试条件**：同一图片（`assets/bus.jpg`） | 同一模型（YOLO11x） | 相同参数（conf=0.25, iou=0.7, imgsz=640）

**验证结论**：Go与Python检测结果高度一致，5个目标全部成功匹配，平均IoU 0.9964，归一化坐标最大偏差<0.003。

| 验证维度 | 结果 | 结论 |
|----------|------|------|
| 检测数量 | 5个目标全部匹配 | 一致 |
| 类别 | 无类别错误 | 一致 |
| 边界框 | 平均IoU 0.9964 | 高度一致 |
| 置信度 | 差异<±0.007 | 高度一致 |

> 验证过程中发现并修复了 Go 端的变量遮蔽（Variable Shadowing）Bug 及 Windows 编码兼容性问题，详见 [test/compare/README.md](test/compare/README.md)。

### 测试程序与论文实验的对应

全部 48 个标准化测试程序位于 `test/benchmark/`，每个均独立可运行，**直接调用 ONNX Runtime 原生 API** 以确保测量纯净度，不依赖 engine 包的封装层。

---

## 🧪 性能测试

本项目包含完整的 **48 个标准化测试程序**，用于比较 Go 和 Python 作为主机语言对 ONNX Runtime 推理性能的影响。

**测试程序构成**：
- **正式测试程序**：48个（24 Go + 18 Python + 2 消融 + 1 72h + 1 C API + 2 Arena消融）
- **辅助程序**：4个（统计分析、图表生成、批量验证、结果对比）
- **Example 示例**：2个

### 快速运行所有测试

**一键运行全部 41 个测试任务（推荐）**：

```bash
# Windows 系统
cd test
run_all_tests_complete.bat
```

该脚本会自动执行以下操作：
- ✅ 运行全部 48 个标准化测试程序
- ✅ 自动检查 Go/Python 环境
- ✅ 自动生成所有图表（PDF + PNG 格式）
- ✅ 自动保存测试结果到 `results/` 目录
- ✅ 自动生成测试摘要报告

测试完成后，查看详细报告：
- 📄 `results/` - 所有测试结果和图表
- 📄 `test/测试规范与性能分析综合报告.md` - 完整测试规范与结果分析

### 测试规范

本项目遵循核心期刊标准的测试规范，确保测试结果的科学性和可复现性。

#### 核心测试原则

**P0 原则（最重要）**：只比较"执行语义"，不比较"API 便利性"
- 不比较 Go 的 AdvancedSession 优势
- 不比较 Python 的高级封装
- 只比较：ORT CPUExecutionProvider + 默认执行路径

**P1 原则（公平性）**
- 相同模型（YOLO11x）
- 相同 ONNX Runtime 版本（1.23.2）
- 相同 Execution Provider（CPUExecutionProvider）
- 相同线程配置（intra_op_num_threads=12, inter_op_num_threads=1）
- 相同 batch size（1）
- 相同输入数据（固定种子 12345）
- 相同 warmup / runs（10 warmup, 200 runs × 10轮）
- 相同 Session 生命周期策略

**P2 原则（可复现）**
- 所有参数显式写死
- 所有随机数固定 seed
- 所有统计指标明确定义

#### 测试环境

| 项目 | 配置 |
|------|------|
| CPU | Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz |
| CPU核心数 | 12 |
| 总内存 | 15.85 GB |
| 操作系统 | Windows 11 x64 |
| Go 版本 | Go 1.25.4 |
| Python 版本 | Python 3.12.0 |
| ONNX Runtime 版本 | 1.23.2 |

### 48个标准化测试程序清单

| 序号 | 测试程序 | 类别 | 状态 |
|------|----------|------|------|
| 1 | go_baseline | 基准测试 | ✅ 完成 |
| 2 | python_baseline | 基准测试 | ✅ 完成 |
| 3 | go_thread_1 | 线程配置 | ✅ 完成 |
| 4 | go_thread_2 | 线程配置 | ✅ 完成 |
| 5 | go_thread_4 | 线程配置 | ✅ 完成 |
| 6 | go_thread_8 | 线程配置 | ✅ 完成 |
| 7 | python_thread_1 | 线程配置 | ✅ 完成 |
| 8 | python_thread_2 | 线程配置 | ✅ 完成 |
| 9 | python_thread_4 | 线程配置 | ✅ 完成 |
| 10 | python_thread_8 | 线程配置 | ✅ 完成 |
| 11 | go_cold_start | 冷启动 | ✅ 完成 |
| 12 | python_cold_start | 冷启动 | ✅ 完成 |
| 13 | go_long_stability | 稳定性 | ✅ 完成 |
| 14 | python_long_stability | 稳定性 | ✅ 完成 |
| 15 | go_reinforced | 强化测试 | ✅ 完成 |
| 16 | python_reinforced | 强化测试 | ✅ 完成 |
| 17 | go_reinforced_small | 强化测试 | ✅ 完成 |
| 18 | python_reinforced_small | 强化测试 | ✅ 完成 |
| 19 | go_memory_standardization | 内存测试 | ✅ 完成 |
| 20 | python_memory_standardization | 内存测试 | ✅ 完成 |
| 21 | go_cold_start_decomposition | 冷启动分解 | ✅ 完成 |
| 22 | python_cold_start_decomposition | 冷启动分解 | ✅ 完成 |
| 23 | go_session_pool | 并发架构 | ✅ 完成 |
| 24 | go_architecture_comparison | 并发架构 | ✅ 完成 |
| 25 | python_architecture_comparison | 并发架构 | ✅ 完成 |
| 26 | go_concurrent_stress | 并发压力 | ✅ 完成 |
| 27 | go_cpu_monitoring | CPU监控 | ✅ 完成 |
| 28 | python_cpu_monitoring | CPU监控 | ✅ 完成 |
| 29 | go_session_creation | Session创建 | ✅ 完成 |
| 30 | python_session_creation | Session创建 | ✅ 完成 |
| 31 | go_pure_inference | 纯推理 | ✅ 完成 |
| 32 | python_pure_inference | 纯推理 | ✅ 完成 |
| 33 | go_performance_diagnostic | 性能诊断 | ✅ 完成 |
| 34 | go_output_consistency | 输出一致性 | ✅ 完成 |
| 35 | python_output_consistency | 输出一致性 | ✅ 完成 |
| 36 | go_yolo11n_reinforced | 轻模型测试 | ✅ 完成 |
| 37 | python_yolo11n_reinforced | 轻模型测试 | ✅ 完成 |
| 38 | go_advanced_session | 高级Session | ✅ 完成 |
| 39 | python_advanced_session | 高级Session | ✅ 完成 |
| 40 | go_batch_inference | 批量推理 | ✅ 完成 |
| 41 | go_long_stability_enhanced | 增强稳定性 | ✅ 完成 |
| 42 | go_session_pool_ablation | 消融实验 | ✅ 完成 |
| 43 | python_session_pool_ablation | 消融实验 | ✅ 完成 |
| 44 | go_72h_stability | 72h稳定性 | ✅ 完成 |
| 45 | cpp_baseline_benchmark | C API基准 | ✅ 完成 |

### 数据精度规范

所有数据符合 **核心期刊数值处理规范**：
- 中间数据：保留5位小数
- 延迟数据：保留3位小数
- 内存数据：保留2位小数
- 百分比数据：保留2位小数

---

## 📊 性能对比（48个标准化测试程序完整结果）

### 基准测试结果（10轮×200次推理，强化测试数据）

| 指标 | Python | Go | 差异 |
|------|--------|----|------|
| 平均延迟 (ms) | 585.35±6.82 | 737.52±31.70 | Go慢25.99% |
| P50延迟 (ms) | 584.44 | 736.63 | Go慢26.04% |
| P90延迟 (ms) | 593.39 | 780.08 | Go慢31.46% |
| P95延迟 (ms) | 596.09 | 783.49 | Go慢31.44% |
| Peak RSS (MB) | 970.81 | 802.24 | Go低17.36% |
| RSS Drift (MB) | 265.46 | 256.11（Go Heap 7.59 MB） | 均稳定 |

注：本节展示的是强化测试（reinforced）结果，执行 10 轮×200 次推理。同线程配置下（均为intra_op=8），Go与Python延迟几乎相同（695.40 ms vs 694.60 ms，差异<1%），表明表观延迟差异主要来自默认线程配置不同。数据来源：go_reinforced_result.txt、python_reinforced_result.txt。

### 统计分析结果

| 统计指标 | 值 |
|----------|-----|
| 延迟差异 | Go慢25.99%（同线程配置差异<1%），差异具有高度统计显著性（p<0.001，Cohen's d>34） |
| 数据可靠性 | 10轮×200次推理，高统计可靠性 |

### 冷启动性能对比

| 指标 | Python | Go | 差异 |
|------|--------|----|------|
| 冷启动时间 (ms) | 608.03 | 759.25 | Go慢24.87% |
| 稳定状态时间 (ms) | 588.30 | 729.48 | Go慢24.00% |
| 冷启动/稳定比例 | 1.03x | 1.04x | 接近 |
| Peak RSS (MB) | 973.99 | 982.53 | 差异<1% |

### 并发架构性能对比（并发数=4）

三种并发架构对比（**Go架构测试结果**）：

| 架构 | 平均延迟(ms) | 吞吐率(REQ/s) | Peak RSS(MB) |
|------|-------------|---------------|--------------|
| Unsafe Shared | 3515.54 | 1.132 | 2216.02 |
| Mutex Shared | 2067.90 | 0.483 | 607.10 |
| **Session Pool** | 3458.10 | 1.146 | 2243.82 |

**说明**：数据来自 `results/go_architecture_comparison.txt`（并发数=4）。Session Pool在资源隔离与稳定性方面具有明显优势。Mutex Shared延迟最低但吞吐量受限于串行化。

### Session Pool扩展性（Python架构测试结果，数据来自python_architecture_comparison.txt）

| 池大小 | 平均延迟(ms) | 吞吐率(REQ/s) | Peak RSS(MB) |
|--------|-------------|---------------|--------------|
| 1 | 2068.97 | 0.48 | 968.13 |
| 2 | 2372.95 | 0.84 | 1517.87 |
| 4 | 3480.81 | 1.15 | 2647.16 |
| 6 | 5181.86 | 1.15 | 3719.60 |
| 8 | 7228.00 | 1.10 | 4826.62 |
| 12 | 11752.82 | 1.01 | 7001.25 |

注：以上为Python Session Pool的实际测试结果（来自 `results/python_architecture_comparison.txt`），线程数=池大小=并发数。吞吐量随池大小增加先升后稳，延迟显著上升（CPU过度订阅）。

### 批量验证性能演进（真实图片端到端，YOLO11x，100张）

以下数据来自 `test/batch_verify/` 工具，覆盖完整管线（图片解码 → 预处理 → 推理 → NMS → 结果输出），更贴近生产环境。

| 版本 | 配置 | 耗时 | FPS | 平均延迟 | P50 | P99 | 内存 | 效率 |
|------|------|-----------|-----|----------|-----|-----|------|------|
| v1 (Bug) | pool=12+, intra=12 | — | — | 35085ms | 23520ms | 173669ms | 294MB | 29% |
| v2 (阻塞池) | pool=12, intra=1 | — | — | 13922ms | 14314ms | 16548ms | 239MB | 77% |
| v3 (4×3) | pool=4, intra=3 | — | — | 6173ms | 6246ms | 10159ms | 93MB | 88% |
| v4 (6×2) | pool=6, intra=2 | — | — | 7746ms | 7716ms | 11994ms | 124MB | 93% |
| **v5★ (2×6)** | **pool=2, intra=6** | 50张 | — | 3062ms | 3082ms | 3261ms | ~62MB | — |
| **v5-stress (2×6)** | **pool=2, intra=6, 4 workers** | 5000张 | — | 3108ms | 3108ms | 3620ms | 77MB峰 | 93% |

**效率** = 实际 FPS / 理论极值（单Session基准约738ms/张）

#### Bug 根因分析

v1 存在两个并发 Bug：

1. **池泄漏**（`sessionPool.get()` 用了 `select + default` 非阻塞模式）：池空时绕过阻塞等待，直接创建新 Session，12 个 worker 最终创建了 12+ 个 Session，远超池容量。

2. **CPU 过度订阅**（未设置 `SetIntraOpNumThreads`）：ONNX Runtime 默认每个 Session 用满全部 CPU 核心。12+ Session × 12 核 = 144+ 线程争抢 12 核，单推理从 0.74s 膨胀到 35s（47 倍退化）。

#### 修复路径

| 修复 | 内容 | 效果 |
|------|------|------|
| 修复 1 (v1→v2) | `get()` 改为阻塞等待，预创建满池 Session | P50延迟从23520ms降至14314ms（降39%），内存从294MB降至239MB |
| 修复 2 (v2→v3) | 设置 `intra_op = CPU/池`，自动线程分配 | P50延迟从14314ms降至6246ms（降56%），内存从239MB降至93MB |
| 优化 3 (v3→v4) | 增加池大小从 4→6，线程从 3→2 | 效率从88%升至93%，内存从93MB升至124MB |
| 调优 4 (v4→v5) | 大模型优先保障线程数：pool=2, intra=6 | 单Session线程充足，推理快，内存仅~62MB |

#### v3/v4/v5 取舍分析

| 维度 | v3 (4×3) | v4 (6×2) | v5★ (2×6) | 原因 |
|------|---------|---------|----------|------|
| 单推理速度 | 较快 (intra=3) | 较慢 (intra=2) | 最快 (intra=6) | 线程越多线性代数越快 |
| 并发度 | 4路 | 6路 | 2路 | 更多Session并行 |
| 最终吞吐 | 88%效率 | 93%效率 | 最优 | v5单Session线程充足，推理极快 |

**结论（50张快速测试）**：大模型优先保障单Session线程数，小模型才可多路并发。

> **注（2000张参数扫描修正）**：50 张快速测试（v5）因样本过小、内存采样不充分，曾将 `pool=2/intraOp=6` 判为最优；系统性参数扫描（`pool=1/2 × intraOp=1–6` 共 12 组、每组 2000 张真实监控帧，详见 `test/测试规范与性能分析综合报告.md` §5.5.6 与 `test/测试程序与图表生成程序完整清单.md` §2.5.3）表明 `pool=2/intraOp=3`（及 2×4）在纯推理延迟（1457 vs 2796 ms）与峰值内存（1171 vs 2363 MB）上均显著优于 2×6，最终推荐配置修正为 **pool=2/intraOp=3**（与论文 §5.2 一致）。

#### 5000 张稳定性验证（v5-stress）

`pool=2, intraOp=6, workers=4`，连续处理 5000 张真实监控图片，每 500 张分段统计：

| 段 | 图片范围 | FPS | P50 | P99 | 说明 |
|----|---------|-----|-----|-----|------|
| 1 | 1-500 | ~1.26 | 3082ms | 3531ms | 冷启动后稳定 |
| 2-9 | 501-4500 | ~1.26 | ~3112-3166ms | ~3595-3947ms | 平稳运行 |
| 10 | 4501-5000 | ~1.26 | 3108ms | 3620ms | 收尾无退化 |

- **总耗时**：约66分钟
- **P50 跨段波动**：仅84ms（3082-3166ms）
- **内存**：运行期间20-47 MB区间波动，峰值77 MB，无泄漏
- **稳定性判定**：✅ 通过 — 10段FPS完全相同，无渐进式性能退化，5000/5000全部成功

#### 与论文基准的关系

论文单 Session 基准测试（`intra_op=12`）是理论上限。批量验证 v5-stress 在端到端真实场景下达到了约1.26 FPS的稳定吞吐，其中：

- 纯推理：约738ms/张（Go baseline单Session）
- 图片 I/O + 预处理 + NMS + 文件写入：约54ms/张额外开销

**论文成果完全成立**：Session Pool架构在真实场景下以93%效率运行，5000张稳定性测试66分钟0失败，性能完全平坦。

---

### 72小时超长时间稳定性测试（已完成）

验证Go/Python单Session接口（poolSize=1, intraOp=12，同口径）在连续运行中的稳定性。

**Go 72h 结果**（2026-06-30 至 2026-07-03）：

| 指标 | 值 |
|------|-----|
| 总运行时间 | 72.00小时 |
| 总推理次数 | 433,652 |
| 平均延迟 | 595.24 ms |
| P50/P99延迟 | 594.45 / 685.24 ms |
| 起始PM / 结束PM | 575.81 / 585.76 MB |
| PM漂移 | +9.95 MB（0.14 MB/h） |
| 错误数 | 0 |

**Python 72h 结果**（2026-07-03 至 2026-07-06）：

| 指标 | 值 |
|------|-----|
| 总运行时间 | 72.00小时 |
| 总推理次数 | 238,675 |
| 平均延迟 | 585.58 ms |
| P50/P99延迟 | 582.47 / 613.61 ms |
| 起始RSS / 结束RSS | 965.21 / 984.23 MB |
| RSS漂移 | +19.02 MB（0.26 MB/h） |
| 错误数 | 0 |

**Go vs Python 同口径对比**（均为单Session，intraOp=12）：

| 指标 | Go | Python | 差异 |
|------|-----|--------|------|
| 平均延迟 | 595.24 ms | 585.58 ms | +1.6%（同线程配置下差距极小） |
| 内存漂移 | 0.14 MB/h | 0.26 MB/h | 两者均≈0，无内存泄漏 |

**结论**：72h稳定性测试Go与Python均零错误完成，内存漂移几乎为零，同intraOp=12下延迟差距仅1.6%。数据文件：`results/go_stability_72h_result.json`、`results/python_stability_72h_result.txt`。

---

### 统计分析（t检验，95%置信区间，Cohen's d）

基于10轮独立重复实验（每轮200次推理）的统计分析：

| 对比组 | Go均值 | Python均值 | 差异 | 结论 |
|--------|--------|-----------|------|------|
| YOLO11x | 737.52 ms | 585.35 ms | Go慢25.99% | 同线程配置差异<1% |
| YOLO11n | 40.73 ms | 37.10 ms | Go慢8.9% | 轻量级模型差异更小 |

数据文件：`results/statistical_analysis.json`。

---

### Session Pool 消融实验（64组配置）

系统研究 Pool Size 和 IntraOp 线程数对性能的影响。Go和Python各32组配置（YOLO11x 16组 + YOLO11n 16组），每组492次推理：

| 关键发现 | 详情 |
|----------|------|
| YOLO11x最优配置（消融） | pool=4, intraOp=2: 延迟3137.5ms, 吞吐1.272 REQ/s |
| YOLO11n最优配置（消融） | pool=4, intraOp=4: 延迟175.4ms, 吞吐22.646 REQ/s |
| 大模型最优配置（实际） | pool=2/intraOp=3（2路并发Run()部署，详见论文 §5.2），避免CPU过度订阅 |
| 小模型最优配置（实际） | 大池大小+多并发，充分利用多核 |
| CPU过度订阅 | 池大小增大→延迟上升3-4倍，吞吐不增反降 |

数据文件：`results/go_session_pool_ablation.json`, `results/python_session_pool_ablation.json`。

---

### C API 基准测试（ONNX Runtime C API）

使用原生C API直接调用ONNX Runtime，测量纯ORT推理性能基线：

| 指标 | 值 |
|------|-----|
| 推理延迟 | 873.57 ms（2000次推理） |
| 标准差 | 159.55 ms |
| P50延迟 | 813.48 ms |
| P99延迟 | 1486.00 ms |
| 吞吐量 | 1.145 REQ/s |

**结论**：C API RSS为504.78 MB，低于Go（798.50 MB）和Python（967.11 MB），说明C API内存开销最小。

数据文件：`results/cpp_baseline_result.json`。

---

## 🏆 核心结论

基于 **48个标准化测试程序**（72h已完成）的完整分析：

| 对比维度 | 结论 | 统计显著性 |
|----------|------|------------|
| **延迟性能** | Python优于Go（585 vs 738 ms），但同线程配置差异<1% | 10轮×200次，高可靠性 |
| **内存效率** | Go单实例Peak RSS约17%低于Python（799 vs 967 MB） | 多次测量一致 |
| **冷启动** | 两者冷启动/稳定比例接近（1.03-1.04x），差异不大 | 冷启动分解测试 |
| **并发性能** | Session Pool优势在资源隔离和CPU利用率梯度，非吞吐量 | 架构对比+CPU监控 |
| **72h稳定性** | Go 72h: PM漂移0.14 MB/h, 433,652次; Python 72h: RSS漂移0.26 MB/h, 238,675次; 均零错误，72h稳定运行 | 72h测试已完成 |
| **C API基线** | C API延迟873.57 ms（2000次样本），P50=813.48 ms | 2000次大样本 |

### 工程实践建议

| 应用场景 | 推荐方案 | 理由 |
|----------|----------|------|
| **延迟敏感场景** | Python + 高intraOp线程 | Python在延迟上约有25%优势（默认配置），同线程配置差异<1% |
| **内存受限/边缘计算** | Go + 合理池大小 | Go单实例Peak RSS约17%低于Python（799 vs 967 MB） |
| **高并发推理** | Go/Python + Session Pool | Session Pool提供资源隔离和CPU利用率梯度优势 |
| **资源监控** | 关注RSS漂移和CPU利用率 | Go PM漂移仅-1.22 MB（10分钟），Python RSS漂移-0.02 MB（10分钟） |

### 文档索引

- 📄 [测试规范与性能分析综合报告](test/测试规范与性能分析综合报告.md) - 完整测试规范与48个标准化测试程序结果（含Go/Python 72h稳定性、统计分析、消融实验、C基准）
- 📄 [测试程序与图表生成程序完整清单](test/测试程序与图表生成程序完整清单.md) - 所有测试程序和推理次数统计
- 📄 [图表生成说明](test/charts/README.md) - 图表生成与使用说明（含期刊规范图表）
- 📄 `results/statistical_analysis.json` - 统计分析结果（t检验/95%CI/Cohen's d）
- 📄 `results/go_session_pool_ablation.json` - Go消融实验原始数据
- 📄 `results/python_session_pool_ablation.json` - Python消融实验原始数据
- 📄 `results/go_stability_1h_result.json` - Go 1h稳定性结果（单Session，同口径）
- 📄 `results/python_stability_1h_result.json` - Python 1h稳定性结果（单Session，同口径）
- 📄 `results/cpp_baseline_result.json` - C API基准测试结果

---

## 📋 支持的类别（80个COCO类别）

支持包括人、车、动物、家具、电器等在内的80个常见物体类别的检测，并提供中文标签显示。

- 人员 (person)
- 交通工具：汽车(car)、摩托车(motorcycle)、飞机(airplane)、公交车(bus)、火车(train)、卡车(truck)、船(boat)等
- 动物：鸟(bird)、猫(cat)、狗(dog)、马(horse)、牛(cow)、大象(elephant)等
- 家具用品：椅子(chair)、沙发(couch)、盆栽(potted plant)、床(bed)等
- 电子设备：电视(tv)、笔记本电脑(laptop)、鼠标(mouse)、遥控器(remote)等
- 食物：香蕉(banana)、苹果(apple)、热狗(hot dog)、披萨(pizza)等
- 以及其他50多个常用类别

## 🚀 性能优化

- 多线程并发处理图像
- 检测器池机制，复用模型会话
- 高效的内存管理和垃圾回收
- ONNX Runtime硬件加速支持
- 图像对象池，减少内存分配
- 批量任务处理，减少上下文切换开销
- 矩形缩放，提高推理速度
- 测试时增强(TTA)，提高检测精度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📄 许可证

MIT License

## 🙏 致谢

- [ultralytics/yolov11](https://docs.ultralytics.com/models/yolo11/) - YOLOv11 模型
- [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) - Go语言ONNX Runtime绑定
- [Go编程语言](https://go.dev/) - Go语言开发
- 人工智能后面的所有人类, 感谢所有开源项目提供的帮助

---

**版本**：v3.5  
**更新日期**：2026-06-30  
**测试程序**：48 个标准化测试程序（24 Go + 18 Python + 2 消融 + 1 72h + 1 C API + 2 Arena消融）  
**测试数据**：累计逾80万次推理（48个标准化程序约77.8万次（含72h稳定性测试Go 433,652 + Python 238,675）+ P3外部效度验证79,550次）