# 🚀 YOLO11/YOLOv8x Go 目标检测器（支持中文标签）

一个基于 **ONNX Runtime** 和 **YOLO11/YOLOv8x** 的轻量级目标检测工具，使用 Go 语言编写，支持中文标签显示、多平台（Windows/macOS/Linux）。

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
│   ├── charts/                               # 图表文件
│   │   ├── cold_start_factor.png              # 冷启动因子分析
│   │   ├── cold_start_vs_stable.png           # 冷启动与稳定状态对比
│   │   ├── inference_flow.png                # 推理流程图表
│   │   ├── latency_boxplot.png               # 延迟分布箱线图
│   │   ├── memory_comparison.png             # 内存使用对比
│   │   ├── rss_curve.png                     # 内存使用曲线
│   │   ├── thread_config_avg_latency.png      # 线程配置平均延迟
│   │   ├── thread_config_latency_distribution.png  # 线程配置延迟分布
│   │   ├── thread_config_memory_usage.png     # 线程配置内存使用
│   │   ├── thread_config_speedup.png          # 线程配置加速比
│   │   └── yolo_evolution.png                # YOLO演进图表
│   ├── cold_start_comparison.pdf               # 冷启动时间对比图表
│   ├── env_check_result.txt                   # 环境检查结果
│   ├── go_advanced_session_supplementary.txt   # Go AdvancedSession补充测试结果
│   ├── go_baseline_detailed_log.txt            # Go基准测试详细日志
│   ├── go_baseline_latency_data.txt            # Go基准测试延迟数据
│   ├── go_baseline_result.txt                  # Go基准测试结果
│   ├── go_cold_start_detailed_log.txt          # Go冷启动测试详细日志
│   ├── go_cold_start_result.txt                # Go冷启动测试结果
│   ├── go_long_stability_result.txt            # Go长时间稳定性测试结果
│   ├── go_rss_curve.csv                        # Go内存使用曲线数据
│   ├── go_thread_1_detailed_log.txt            # Go线程配置1测试详细日志
│   ├── go_thread_1_result.txt                  # Go线程配置1测试结果
│   ├── go_thread_2_detailed_log.txt            # Go线程配置2测试详细日志
│   ├── go_thread_2_result.txt                  # Go线程配置2测试结果
│   ├── go_thread_4_detailed_log.txt            # Go线程配置4测试详细日志
│   ├── go_thread_4_result.txt                  # Go线程配置4测试结果
│   ├── go_thread_8_detailed_log.txt            # Go线程配置8测试详细日志
│   ├── go_thread_8_result.txt                  # Go线程配置8测试结果
│   ├── go_thread_config_comprehensive.txt      # Go线程配置综合结果
│   ├── latency_boxplot.pdf                     # 延迟分布箱线图
│   ├── model_md5.txt                          # 模型MD5校验结果
│   ├── python_baseline_detailed_log.txt        # Python基准测试详细日志
│   ├── python_baseline_latency_data.txt        # Python基准测试延迟数据
│   ├── python_baseline_result.txt              # Python基准测试结果
│   ├── python_cold_start_detailed_log.txt      # Python冷启动测试详细日志
│   ├── python_cold_start_result.txt            # Python冷启动测试结果
│   ├── python_long_stability_result.txt        # Python长时间稳定性测试结果
│   ├── python_rss_curve.csv                    # Python内存使用曲线数据
│   ├── python_thread_1_detailed_log.txt        # Python线程配置1测试详细日志
│   ├── python_thread_1_result.txt              # Python线程配置1测试结果
│   ├── python_thread_2_detailed_log.txt        # Python线程配置2测试详细日志
│   ├── python_thread_2_result.txt              # Python线程配置2测试结果
│   ├── python_thread_4_detailed_log.txt        # Python线程配置4测试详细日志
│   ├── python_thread_4_result.txt              # Python线程配置4测试结果
│   ├── python_thread_8_detailed_log.txt        # Python线程配置8测试详细日志
│   ├── python_thread_8_result.txt              # Python线程配置8测试结果
│   ├── python_thread_config_comprehensive.txt  # Python线程配置综合结果
│   ├── rss_curve.pdf                          # 内存使用曲线图表
│   └── thread_config_comparison.pdf            # 线程配置性能对比图表
├── test/             # 测试脚本和数据
│   ├── benchmark/    # Go基准测试
│   │   ├── cold_start_benchmark.go             # Go冷启动测试
│   │   ├── go_baseline_minimal.go              # Go基准测试
│   │   ├── go_long_stability.go                # Go长时间稳定性测试
│   │   ├── thread_config_benchmark.go          # Go线程配置测试
│   │   └── go_advanced_session_supplementary.go # Go AdvancedSession补充测试
│   ├── charts/       # 图表生成脚本
│   │   ├── generate_charts_png.py              # 生成PNG格式图表
│   │   ├── generate_cold_start_and_thread_charts.py  # 生成冷启动和线程配置图表
│   │   ├── generate_latency_boxplot.py         # 生成延迟箱线图
│   │   ├── generate_main_charts.py             # 生成主要图表
│   │   └── plot_rss_curve.py                    # 生成RSS内存曲线
│   ├── data/         # 测试数据
│   │   └── input_data.bin                       # 统一输入数据文件
│   ├── python/       # Python相关测试
│   │   ├── python_baseline.py                   # Python基准测试
│   │   ├── python_baseline_supplementary.py     # Python Baseline补充测试
│   │   ├── python_cold_start_benchmark.py       # Python冷启动测试
│   │   ├── python_long_stability.py             # Python长时间稳定性测试
│   │   └── python_thread_config_benchmark.py    # Python线程配置测试
│   ├── check_environment.py                     # 环境检查脚本
│   ├── env_check.py                             # 环境检查脚本
│   ├── generate_input_data.py                   # 生成统一输入数据
│   ├── generate_model_md5.py                   # 生成模型MD5校验
│   └── 测试规范与性能分析综合报告.md             # 测试规范与性能分析综合报告
├── third_party/      # 第三方依赖
│   ├── onnxruntime.dll  # ONNX Runtime库
│   ├── yolo11x.onnx     # YOLO11x模型
│   └── yolov8x.onnx     # YOLOv8x模型
├── go.mod            # Go模块文件
└── go.sum            # Go依赖校验文件
```

## 🧪 性能测试

本项目包含完整的性能测试程序，用于比较 Go 和 Python 作为主机语言对 ONNX Runtime 推理性能的影响。

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
- 相同线程配置（intra_op_num_threads=4, inter_op_num_threads=1）
- 相同 batch size（1）
- 相同输入数据（固定种子 12345）
- 相同 warmup / runs（10 warmup, 100 runs）
- 相同 Session 生命周期策略

**P2 原则（可复现）**
- 所有参数显式写死
- 所有随机数固定 seed
- 所有统计指标明确定义

#### 测试环境

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i5-10400（2.9GHz 基准频率，4.3GHz 最大睿频，6核心12线程） |
| 内存 | 16 GB |
| 操作系统 | Windows 11 x64 |
| Go 版本 | Go 1.25 |
| Python 版本 | Python 3.12.x |
| ONNX Runtime 版本 | 1.23.2 |

### 测试程序列表

#### Go 测试程序
1. `go_baseline_minimal.go` - Go 基准测试
2. `cold_start_benchmark.go` - Go 冷启动测试
3. `thread_config_benchmark.go` - Go 线程配置测试
4. `go_long_stability.go` - Go 长时间稳定性测试
5. `go_advanced_session_supplementary.go` - Go AdvancedSession 补充测试

#### Python 测试程序
1. `python_baseline.py` - Python 基准测试
2. `python_cold_start_benchmark.py` - Python 冷启动测试
3. `python_thread_config_benchmark.py` - Python 线程配置测试
4. `python_long_stability.py` - Python 长时间稳定性测试
5. `python_baseline_supplementary.py` - Python Baseline 补充测试

### 图表生成脚本

#### PDF 图表
- `test/charts/generate_latency_boxplot.py` - 生成延迟箱线图
- `test/charts/plot_rss_curve.py` - 生成 RSS 内存曲线
- `test/charts/generate_cold_start_and_thread_charts.py` - 生成冷启动和线程配置图表

#### PNG 图表
- `test/charts/generate_charts_png.py` - 生成 PNG 格式图表
- `test/charts/generate_main_charts.py` - 生成主要图表

### 测试文档

- `test/测试规范与性能分析综合报告.md` - 测试规范与性能分析综合报告

## 📊 性能对比

### 推理性能（YOLO11x）

| 实现语言 | Avg (ms) | P50 (ms) | P90 (ms) | P99 (ms) | 相对性能 |
|---------|----------|----------|----------|----------|----------|
| Python  | 952.234 | 950.797 | 986.161 | 1026.292 | 1.00× |
| Go      | 903.297 | 902.611 | 915.386 | 970.881 | 0.95× |

**性能差异**：Go 比 Python 快 5.13%

### 内存使用（YOLO11x）

| 实现语言 | Start RSS (MB) | Peak RSS (MB) | Stable RSS (MB) | RSS Drift (MB) | 内存效率 |
|---------|---------------|---------------|----------------|----------------|----------|
| Python  | 292.78        | 549.44        | 549.39         | 256.61         | 1.00× |
| Go      | 62.29         | 62.61         | 62.16          | -0.13          | 8.81× |

**内存效率**：Go 内存使用仅为 Python 的 1/8.81

### 长时间稳定性测试（10分钟）

| 指标 | Go | Python |
|------|----|--------|
| 测试时长 | 10m2s | 601秒 |
| 推理次数 | 599 | 294 |
| 推理频率 | 1.00 次/秒 | 0.49 次/秒 |
| 平均推理时间 | 896.190 ms | 1042.995 ms |
| 初始 RSS | 62.66 MB | 554.22 MB |
| 最终 RSS | 62.12 MB | 554.46 MB |
| RSS Drift | -0.54 MB | 0.24 MB |
| RSS 波动范围 | 0.81 MB (1.30%) | 0.28 MB (0.05%) |

### 线程配置测试结果

| 线程数 | Go 延迟 | Python 延迟 | 差异 | 优势 |
|--------|---------|------------|------|------|
| 1 | 899.022 ms | 2258.219 ms | -60.2% | **Go** |
| 2 | 898.007 ms | 1308.488 ms | -31.4% | **Go** |
| 4 | 896.928 ms | 947.116 ms | -5.3% | **Go** |
| 8 | 897.169 ms | 734.746 ms | +22.1% | **Python** |

**关键发现**：
- Go 在 1-4 线程配置下性能优于 Python
- Python 在 8 线程配置下性能优于 Go（多线程优化）
- Go 内存使用始终优于 Python（8.6-8.9倍）
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
