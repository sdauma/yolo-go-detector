# Python ↔ Go YOLO 检测结果一致性对比

对同一张图片、同一模型（YOLO11x）、相同参数（conf=0.25, iou=0.7, imgsz=640），分别用 Python（Ultralytics）和 Go（ONNX Runtime 原生 API）进行端到端推理，逐项对比检测结果。

## 快速使用

```bash
# 1. 编译 Go 对比工具
cd test/compare
go build -o compare.exe .

# 2. 运行对比
python compare.py
```

## 工具说明

| 文件 | 作用 |
|------|------|
| `main.go` | Go 端检测器：加载图片 → ONNX Runtime 推理 → NMS 后处理 → 输出 YOLO 格式 txt + 标注图片 |
| `compare.py` | 对比脚本：调用 Python YOLO 和 Go 检测器 → 解析结果 → IoU 匹配 → 输出对比报告 |
| `bus_go_detections.txt` | Go 检测结果（YOLO 归一化格式：class cx cy w h conf） |
| `bus_go_result.jpg` | Go 检测标注图片 |

## 最终结论（2026-06-04）

**检测结果完全一致**：Go 和 Python 均检出 **5 个目标**（1 巴士 + 4 人员），全部 5 对匹配，平均 IoU **0.9964**。

### 逐目标对比

| 目标 | Go (cx,cy,w,h) | Python (cx,cy,w,h) | IoU | 坐标最大偏差 | Go conf | Python conf | conf 差异 |
|------|----------------|---------------------|-----|-------------|---------|-------------|-----------|
| 巴士 | (0.5038,0.4479,0.9784,0.4678) | (0.5041,0.4478,0.9779,0.4681) | 0.9987 | 0.0004 | 0.9428 | 0.9397 | +0.0032 |
| 人员1 | (0.1834,0.6041,0.2429,0.4636) | (0.1837,0.6043,0.2434,0.4640) | 0.9976 | 0.0004 | 0.9196 | 0.9229 | -0.0033 |
| 人员2 | (0.9128,0.5897,0.1741,0.4521) | (0.9127,0.5885,0.1741,0.4546) | 0.9961 | 0.0025 | 0.9164 | 0.9162 | +0.0002 |
| 人员3 | (0.3505,0.5863,0.1496,0.4206) | (0.3505,0.5864,0.1496,0.4207) | 0.9989 | 0.0001 | 0.9119 | 0.9053 | +0.0067 |
| 人员4 | (0.0490,0.6586,0.0974,0.2983) | (0.0488,0.6589,0.0970,0.2990) | 0.9907 | 0.0007 | 0.8577 | 0.8508 | +0.0069 |

- 坐标偏差均在 **0.003 以内**（归一化坐标），对应 810×1080 原图上 < 3 像素
- 置信度差异 ±0.007 以内，由浮点运算精度和预处理细微差异导致
- **不存在漏检、误检、类别错误**

## 已修复的关键 Bug

### 变量遮蔽（Variable Shadowing）

**症状**：Go 仅检出 2 个目标（vs Python 5 个）

**根因**：`processOutput()` 函数中，外层的 `origW`/`origH`（原图尺寸）被内层局部变量 `origW := w / si.ScaleX` 遮蔽，导致 `clamp()` 上限使用了错误的参考尺寸，使大量检测框被裁剪出界。

**修复**：将内层局部变量重命名为 `boxW`/`boxH`

```go
// ❌ Bug: 局部变量遮蔽了外层的图像尺寸变量
origW := w / si.ScaleX   // 这是框宽，不是图像宽度
origH := h / si.ScaleY
x1 = clamp(x1, 0, float32(origW)-1)  // 用框宽作为 clamp 上限 → 错误裁剪

// ✅ 修复: 使用独立的变量名
boxW := w / si.ScaleX
boxH := h / si.ScaleY
x1 = clamp(x1, 0, float32(origW))    // 正确使用图像宽度作为 clamp 上限
y1 = clamp(y1, 0, float32(origH))
```

### 其他修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `undefined: image.RGBA` 编译错误 | Go 标准库中 `image.RGBA` 不存在，应使用 `color.RGBA` | 导入 `"image/color"`，改用 `color.RGBA{...}` |
| Python `subprocess` 编码错误 | Windows 默认 GBK 无法解码 Go 的 UTF-8 输出 | `encoding='utf-8', errors='replace'` |
| Python 标注文件路径错误 | YOLO ONNX 模式输出路径与 PyTorch 模式不同 | 同时检测 `predict/labels/` 和 `labels/` 两个路径 |

## 与现有测试的关系

本工具补充了项目测试体系中缺失的一环：

| 测试类型 | 已有工具 | 本工具补充 |
|----------|----------|-----------|
| 模型输出张量一致性 | `go_output_consistency` / `python_output_consistency` | ✅ 端到端检测结果（类别+坐标+置信度） |
| 延迟/内存/并发 | 48 个标准化 Benchmark | — |

> 注：另有后续补充程序（`thread_config_benchmark_yolo11n.go`、`python_thread_config_yolo11n_benchmark.py`、`go_session_lifecycle_repro.go`、`go_session_pool_fault_injection.go` 等）未计入上述 48，分别支撑论文 §4.3 / §4.1 补充结论与 §4.6.1 / §5.1 / §5.2 / §7 的 Session Pool 故障隔离与自动重建实验验证。
| **跨语言检测一致性** | — | ✅ `compare` + `batch_verify` |

## 参数

```bash
# Go 检测器参数
go run . -model ../../third_party/yolo11x.onnx -img ../../assets/bus.jpg -conf 0.25 -iou 0.7 -size 640
```