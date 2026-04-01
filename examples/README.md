# Go-YOLO 示例程序

本目录包含使用 `engine` 和 `detector` 包的完整示例程序。

## 示例列表

### 基础示例

1. **01_basic_session_pool.go** - Session Pool 基础使用
   - 展示如何创建和使用 Session Pool
   - 演示并发推理
   - 适合快速了解 Session Pool 的工作原理

2. **02_batch_inference.go** - 批量推理引擎
   - 使用 BatchInferenceEngine 处理批量数据
   - 展示性能统计和池监控
   - 适合批处理场景

### 应用示例

3. **03_realtime_detector.go** - 实时检测器
   - 使用 `engine.BatchInferenceEngine` 模拟实时检测场景
   - 展示任务提交和结果处理的完整流程
   - 适合实时处理场景参考

**注意**: `detector` 包当前为框架代码，核心检测方法尚未实现。实际项目建议直接使用 `engine` 包构建应用。

## 运行方法

### 前置条件

1. 确保已安装 Go 1.21+
2. 确保 `third_party/` 目录下有：
   - `onnxruntime.dll` (Windows)
   - `yolo11x.onnx`

### 运行示例

所有示例程序都已通过测试验证。

```bash
# 运行 Session Pool 示例（4 个并发 worker）
go run examples/01_basic_session_pool.go

# 运行批量推理示例（10 个批处理任务）
go run examples/02_batch_inference.go

# 运行实时检测器（5 次实时检测模拟）
go run examples/03_realtime_detector.go
```

## 与 test/ 的区别

- **examples/**: 展示如何使用 engine/detector 构建应用
- **test/**: 科学实验，直接调用底层 API 确保公平对比

**不要混用！** examples/ 是工程代码，test/ 是科研代码。
