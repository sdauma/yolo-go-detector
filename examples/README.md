# Engine API 示例

本目录包含 `engine` 包（论文 §2.3 Session Pool 的工程化实现）的调用示例。

## example.go — BatchInferenceEngine API 演示

- 创建 `BatchInferenceEngine`，提交 10 个推理任务
- 演示 worker 池、task queue、callback 机制
- **注意**：`processTask()` 尚未集成 `engine/postprocess.go`，返回空 `[]BoundingBox{}`
- 完整的端到端检测请参考仓库根目录的 `main.go` + `detector_pool.go`

## 运行

```bash
go run examples/example.go
```

前置条件：`third_party/` 目录下有 `onnxruntime.dll` 和 `yolo11x.onnx`。

## 与 test/ 的区别

- **examples/**: 展示 `engine` 包的 API 用法
- **test/benchmark/**: 论文实验代码，直接调用 ONNX Runtime 原生 API 确保测量纯净度
