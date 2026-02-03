# 发布说明：YOLO11x ONNX 推理系统性能对比

## 版本信息

- **版本**: v2.0.0
- **发布日期**: 2026-02-03
- **项目地址**: https://github.com/sdauma/yolo-go-detector

## 发布内容

本版本包含了 YOLO11x ONNX 模型在 Go 和 Python 中的完整实现，以及严格控制的性能对比实验。主要内容包括：

### 核心功能

- ✅ **跨语言对比**: Go 和 Python 两种实现的完整对比
- ✅ **多种线程配置**: 支持 intra=1/2/4 的线程配置
- ✅ **内存监控**: 提供 RSS 内存占用的实时监控
- ✅ **长时间稳定性测试**: 验证系统的长期运行稳定性
- ✅ **数值一致性验证**: 确保两种实现的输出结果一致

### 新增文件

- **基准测试文件**:
  - `benchmark_go_std_intra1.go` - Go 实现，intra=1
  - `benchmark_go_std_intra2.go` - Go 实现，intra=2
  - `benchmark_go_std_intra4.go` - Go 实现，intra=4
  - `benchmark_go_long_stability.go` - Go 长时间稳定性测试
  - `inference_align.py` - Python 对齐实现

- **内存监控脚本**:
  - `monitor_go_memory.ps1` - Go 内存监控
  - `monitor_python_memory.ps1` - Python 内存监控
  - `monitor_long_stability.ps1` - 长时间稳定性监控

- **依赖配置**:
  - `requirements_inference.txt` - Python 推理依赖

- **结果目录**:
  - `results/` - 存放测试结果文件

## 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/sdauma/yolo-go-detector.git
cd yolo-go-detector
```

### 2. 安装 Go 依赖

确保您的系统已安装 Go 1.25 或更高版本：

```bash
go version
go mod tidy
```

### 3. 安装 Python 依赖

确保您的系统已安装 Python 3.12 或更高版本：

```bash
python --version
pip install -r requirements_inference.txt
```

### 4. 准备模型文件

确保 `third_party/` 目录中包含以下文件：
- `onnxruntime.dll` (Windows) 或相应的库文件
- `yolo11x.onnx` (YOLO11x 模型文件)

## 使用指南

### 1. 运行基准测试

#### Go 基准测试

```bash
# 运行 intra=1 配置
go run benchmark_go_std_intra1.go

# 运行 intra=2 配置
go run benchmark_go_std_intra2.go

# 运行 intra=4 配置
go run benchmark_go_std_intra4.go

# 运行长时间稳定性测试
go run benchmark_go_long_stability.go
```

#### Python 基准测试

```bash
python inference_align.py
```

### 2. 运行内存监控

#### Go 内存监控

```powershell
# 在一个终端中启动内存监控
./monitor_go_memory.ps1

# 在另一个终端中运行基准测试
go run benchmark_go_std_intra4.go
```

#### Python 内存监控

```powershell
# 在一个终端中启动内存监控
./monitor_python_memory.ps1

# 在另一个终端中运行基准测试
python inference_align.py
```

### 3. 查看测试结果

测试结果将保存到以下文件：
- Go 测试结果: `results/go_benchmark_*.txt`
- Python 测试结果: 控制台输出

## 实验复现指南

为了确保实验结果的可复现性，请按照以下步骤进行：

### 1. 硬件与软件环境

- **CPU**: Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz 或类似性能
- **内存**: 16 GB 或更多
- **操作系统**: Windows 11 x64
- **Go**: 1.25
- **Python**: 3.12
- **ONNX Runtime**: 1.23.2

### 2. 实验配置

- **批处理大小**: 1
- **并发流数**: 1
- **Intra-op 线程数**: 1/2/4
- **Inter-op 线程数**: 1
- **预热运行次数**: 10
- **测量运行次数**: 30
- **输入图像**: 810×1080 RGB (assets/bus.jpg)

### 3. 运行顺序

1. 关闭系统中其他占用资源的程序
2. 运行 Go 基准测试 (intra=1/2/4)
3. 运行 Python 基准测试
4. 运行长时间稳定性测试
5. 收集并分析测试结果

## 预期结果

### 性能对比

| 实现语言 | Avg (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|---------|----------|----------|----------|----------|
| Python  | 3382.21  | 3711.85  | 4184.99  | 4590.05  |
| Go      | 1087.35  | 1088.36  | 1207.51  | 1260.42  |

### 内存占用

| 实现语言 | Peak RSS (MB) | Stable RSS (MB) |
|---------|---------------|-----------------|
| Python  | 558.3         | 558.3           |
| Go      | 530.3         | 530.3           |

### 稳定性

Go 实现在长时间运行中表现出稳定的内存使用，无明显内存漂移。

## 故障排除

### 1. 模型加载失败

- 检查 ONNX 模型文件的路径和格式
- 确保 ONNX Runtime 库的版本与模型兼容
- 验证 `third_party/` 目录中是否存在正确的库文件

### 2. 性能测试结果不一致

- 关闭系统中其他占用资源的程序
- 确保按照相同的硬件和软件环境进行测试
- 运行多次测试取平均值，减少随机因素的影响

### 3. 内存监控脚本无法找到进程

- 确保基准测试程序正在运行
- 检查监控脚本中指定的进程名称是否正确
- 尝试手动指定进程 ID

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 引用

如果您使用本项目的代码或结果，请引用我们的论文：

```
@article{yolo-go-python-comparison,
title={面向边缘推理系统的跨语言深度学习推理性能与稳定性分析},
author={Your Name},
year={2026},
journal={Journal Name}
}
```

## 联系方式

- **作者**: Your Name
- **联系方式**: your-email@example.com
- **项目地址**: https://github.com/sdauma/yolo-go-detector

---

**感谢您使用本项目！** 如有任何问题或建议，请随时联系我们。