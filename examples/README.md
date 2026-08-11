# Engine API 示例

本目录包含 `engine` 包（论文 §2.2.3 Session Pool 的工程化实现，GetSession/PutSession 核心流程）的调用示例，以及论文核心贡献之一的"Go 与 Python 推理一致性"验证示例。

## example.go — BatchInferenceEngine API 演示

- 创建 `BatchInferenceEngine`，提交 10 个推理任务
- 演示 worker 池、task queue、callback 机制
- **注意**：`processTask()` 尚未集成 `engine/postprocess.go`，返回空 `[]BoundingBox{}`
- 完整的端到端检测请参考仓库根目录的 `main.go` + `detector_pool.go`

## consistency/ — Go-Python 推理一致性验证

论文 §3.3 声明"Go 侧推理结果与 Python 官方 ONNX Runtime 推理结果高度一致"。本示例展示验证方法：

- 加载 `assets/bus.jpg`，Letterbox 预处理（pool=1, intra_op=6，论文 §5.2 主推荐配置）
- 对 `yolo11x` 与 `yolo11n` 两个模型分别导出**输入张量** `results/go_<model>_input.bin`（float32 小端序，形状 [1,3,640,640] NCHW）与**原始输出张量** `results/go_<model>_output.bin`（[1,84,8400]）
- 打印检测结果（坐标、类别、置信度）

> 验证采用**方案 B（共用输入张量）**：Go 导出其预处理后的输入张量，Python 直接加载该张量作为模型输入，从而隔离两侧预处理链路差异，仅对比 ONNX Runtime 的 Go 绑定与 Python 绑定在**同一输入**下的输出一致性（论文 §3.3 真正声明）。

**验证步骤**：

```bash
# 1. Go 侧（本示例，从仓库根目录运行）——导出 go_<model>_input.bin 与 go_<model>_output.bin
go run ./examples/consistency

# 2. Python 侧——加载 Go 的输入张量推理并自动比对输出（预期 max|diff| < 1e-4）
python test/python/python_output_consistency.py
```

## 运行

```bash
go run examples/example.go
go run ./examples/consistency
```

前置条件：`third_party/` 目录下有 `onnxruntime.dll`（或对应平台动态库）和 `yolo11x.onnx` / `yolo11n.onnx`。

## 与 test/ 的区别

- **examples/**: 展示 `engine` 包的 API 用法与论文成果的端到端示例
- **test/benchmark/**: 论文实验代码，直接调用 ONNX Runtime 原生 API 确保测量纯净度