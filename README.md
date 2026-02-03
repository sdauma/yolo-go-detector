# YOLO11x ONNX 推理系统：Go 与 Python 性能对比

## 项目概述

这是一个用于评估主机语言对 ONNX Runtime 推理性能影响的系统性研究项目，专注于 YOLO11x 模型在 Go 和 Python 中的推理性能与稳定性对比。该项目提供了完整的基准测试框架、内存监控工具和可复现的实验配置，旨在为边缘和生产环境中的深度学习推理系统设计提供参考。

## 研究背景

随着深度学习模型在边缘和工业场景的广泛应用，系统级设计决策（如主机语言选择、内存管理、并发策略）对推理性能和稳定性的影响日益显著。本项目通过严格控制实验条件，量化了 Go 与 Python 作为主机语言对 YOLO11x ONNX 推理系统的影响。

## 功能特点

- ✅ **跨语言对比**：提供 Go 和 Python 两种实现的完整对比
- ✅ **ONNX Runtime**：使用 ONNX Runtime 1.23.2 作为统一推理后端
- ✅ **严格控制**：两种实现使用相同的模型、硬件、配置和预处理流程
- ✅ **全面基准测试**：支持多种线程配置（intra=1/2/4）和长时间稳定性测试
- ✅ **内存监控**：提供 RSS 内存占用的实时监控和分析
- ✅ **数值一致性**：确保两种实现的输出结果数值一致
- ✅ **可复现性**：提供详细的实验配置和环境说明

## 技术架构

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│   图像预处理        │────▶│   ONNX Runtime      │────▶│   后处理与分析      │
│                     │     │   模型推理          │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
        ▲                          ▲                            │
        │                          │                            ▼
┌────────────────┐         ┌────────────────┐           ┌─────────────────────┐
│                │         │                │           │                     │
│  主机语言实现   │         │  YOLO11x ONNX  │           │   性能指标收集       │
│  (Go / Python)  │         │    模型文件     │           │  (延迟、内存、稳定性) │
└────────────────┘         └────────────────┘           └─────────────────────┘
```

## 安装依赖

### 1. Go 语言环境

确保您的系统已安装 Go 1.25 或更高版本：

```bash
go version
```

### 2. Python 环境

确保您的系统已安装 Python 3.12 或更高版本：

```bash
python --version
```

### 3. ONNX Runtime

#### Go 依赖

```bash
go get github.com/yalue/onnxruntime_go
```

#### Python 依赖

```bash
pip install onnxruntime==1.23.2 opencv-python numpy
```

### 4. 其他依赖

#### Go 依赖

```bash
go get github.com/flopp/go-findfont
go get golang.org/x/image/font
go get golang.org/x/image/font/opentype
go get golang.org/x/image/math/fixed
```

## 快速开始

### 1. 准备模型文件

将 YOLO11x 模型转换为 ONNX 格式，并将其放置在 `third_party/` 目录下，命名为 `yolo11x.onnx`。

### 2. 准备测试图像

将测试图像放置在 `assets/` 目录下，例如 `bus.jpg`。

### 3. 运行基准测试

#### Go 基准测试

```bash
# 运行 intra=1 配置
go run benchmark_go_std_intra1.go

# 运行 intra=2 配置
go run benchmark_go_std_intra2.go

# 运行 intra=4 配置
go run benchmark_go_std_intra4.go
```

#### Python 基准测试

```bash
python inference_align.py
```

### 4. 运行内存监控

#### Go 内存监控

```powershell
./monitor_go_memory.ps1
```

#### Python 内存监控

```powershell
./monitor_python_memory.ps1
```

## 核心功能说明

### 1. 图像预处理

使用 letterbox 算法对输入图像进行预处理，保持宽高比并缩放至模型输入尺寸 (640x640)：

```go
// Go 实现
scaleInfo := fillInputTensor(img, inputTensor)

// Python 实现
img, scale, pad_x, pad_y = letterbox(img)
```

### 2. 模型推理

使用 ONNX Runtime 进行模型推理，支持会话复用和显式张量管理：

```go
// Go 实现
session, err := ort.NewAdvancedSession(...)
if err := session.Run(); err != nil {
    panic(err)
}
```

### 3. 性能评估

提供全面的性能指标评估，包括平均延迟、p50/p90/p99 分位数延迟、内存占用和长时间稳定性：

```go
// 计算统计信息
avg := sum / float64(N)
p50 := times[N/2]
p90 := times[int(math.Floor(float64(N)*0.9))]
p99 := times[int(math.Floor(float64(N)*0.99))]
```

## 实验配置

### 硬件与软件环境

| 项目 | 配置说明 |
|------|----------|
| CPU | Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz |
| 内存 | 16 GB |
| 操作系统 | Windows 11 x64 |
| ONNX Runtime | 1.23.2 (CPUExecutionProvider) |
| 模型 | YOLO11x (ONNX, FP32) |
| 主机语言 | Go 1.25 / Python 3.12 |
| 图优化 | 禁用 |

### 推理配置

| 项目 | 设置说明 |
|------|----------|
| 批处理大小 | 1 |
| 并发流数 | 1 |
| Intra-op 线程数 | 1/2/4 |
| Inter-op 线程数 | 1 |
| 会话生命周期 | 单会话复用 |
| 张量生命周期 | 显式销毁 |
| 预热运行次数 | 10 |
| 测量运行次数 | 30 |
| 输入图像 | 810×1080 RGB |
| 预处理 | Letterbox 缩放 640×640, 常数填充 114, HWC→CHW, float32 |

## 项目结构

```
.
├── assets/              # 测试图像和资源文件
├── third_party/         # 第三方依赖和模型文件
├── benchmark_go_std_intra1.go  # Go 基准测试文件 (intra=1)
├── benchmark_go_std_intra2.go  # Go 基准测试文件 (intra=2)
├── benchmark_go_std_intra4.go  # Go 基准测试文件 (intra=4)
├── benchmark_go_long_stability.go  # Go 长时间稳定性测试
├── inference_align.py   # Python 对齐实现
├── monitor_go_memory.ps1  # Go 内存监控脚本
├── monitor_python_memory.ps1  # Python 内存监控脚本
├── README.md            # 项目说明书
├── go.mod               # Go 模块定义
├── go.sum               # Go 依赖校验
└── labels.txt           # 类别标签文件
```

## 实验结果

### 推理性能对比

| 实现语言 | Avg (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|---------|----------|----------|----------|----------|
| Python  | 3382.21  | 3711.85  | 4184.99  | 4590.05  |
| Go      | 1087.35  | 1088.36  | 1207.51  | 1260.42  |

### 内存占用对比

| 实现语言 | Peak RSS (MB) | Stable RSS (MB) |
|---------|---------------|-----------------|
| Python  | 558.3         | 558.3           |
| Go      | 530.3         | 530.3           |

### 长时间稳定性

Go 实现在长时间运行中表现出稳定的内存使用，无明显内存漂移，适合边缘和生产环境部署。

## 使用示例

### 运行基准测试

#### Go 实现

```bash
# 运行标准配置测试 (intra=4, inter=1)
go run benchmark_go_std_intra4.go

# 运行长时间稳定性测试
go run benchmark_go_long_stability.go
```

#### Python 实现

```bash
# 运行对齐的 Python 测试
python inference_align.py
```

### 内存监控

```powershell
# 启动 Go 内存监控
./monitor_go_memory.ps1

# 启动 Python 内存监控
./monitor_python_memory.ps1
```

## 性能优化

- **会话复用**：创建一次会话，多次使用，减少初始化开销
- **显式张量管理**：及时销毁不再使用的张量，避免内存泄漏
- **线程配置**：根据硬件特性调整 intra-op 线程数，平衡并行度和开销
- **内存管理**：Go 的显式内存管理有助于减少内存占用和提高稳定性

## 常见问题

### 1. 模型加载失败

请检查 ONNX 模型文件的路径和格式是否正确，确保 ONNX Runtime 库的版本与模型兼容。

### 2. 性能测试结果不一致

请确保关闭系统中其他占用资源的程序，按照相同的硬件和软件环境进行测试，以确保结果的可复现性。

### 3. 内存监控脚本无法找到进程

请确保基准测试程序正在运行，并且进程名称与监控脚本中指定的名称匹配。

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目！

## 论文引用

如果您使用本项目的代码或结果，请引用我们的论文：

```
@article{yolo-go-python-comparison,
title={面向边缘推理系统的跨语言深度学习推理性能与稳定性分析},
author={Your Name},
year={2026},
journal={Journal Name}
}
```

## 更新日志

### v2.0.0 (2026-02-03)

- 完成 YOLO11x ONNX 模型的 Go 和 Python 实现
- 实现严格控制的性能对比实验
- 添加内存监控和稳定性测试
- 提供完整的实验复现说明
- 更新项目文档和论文支持

---

**作者**: Your Name
**联系方式**: your-email@example.com
**项目地址**: https://github.com/sdauma/yolo-go-detector
