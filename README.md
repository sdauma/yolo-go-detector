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

### 5. 下载模型文件
下载 YOLOv11 模型文件并放置到 `./third_party/` 目录下，导出为 `yolo11x.onnx`。
如无特殊要求请使用默认参数: yolo export model=yolo11x.pt format=onnx imgsz=640 opset=17 ,默认参数下请使用rect=false ,本程序的rect=true仅在导出参数dynamic=True时有意义。

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
go run . -img ./assets/test.jpg -output ./output/result.jpg -conf 0.5
```

批量处理目录中的图像：
```bash
go run . -img ./test_images/ -conf 0.3 -workers 4
```

启用系统文本标注：
```bash
go run . -img ./assets/test.jpg -enable-system-text=true -system-text="智能安全监控系统" -text-location="top-left"
```

## 🏗️ 项目结构

```
yolo-go-detector/
├── main.go           # 主程序入口，包含检测逻辑
├── detector_pool.go  # 检测器池，支持并发处理
├── README.md
├── LICENSE
├── assets/           # 示例图像
├── third_party/      # 第三方依赖（ONNX模型、运行库）
├── test/             # 性能测试程序
├── results/          # 测试结果存储
└── go.mod/go.sum     # Go模块文件
```

## 🧪 性能测试

本项目包含完整的性能测试程序，用于比较 Go 和 Python 作为主机语言对 ONNX Runtime 推理性能的影响。

### 测试目录结构

```
test/
├── benchmark_go_std_intra1.go    # Go 基准测试（intra_op_num_threads=1）
├── benchmark_go_std_intra2.go    # Go 基准测试（intra_op_num_threads=2）
├── benchmark_go_std_intra4.go    # Go 基准测试（intra_op_num_threads=4）
├── benchmark_go_long_stability.go # Go 长时间稳定性测试
├── inference_align.py            # Python 对齐实现
├── monitor_go_memory.ps1         # Go 内存监控脚本
├── monitor_python_memory.ps1     # Python 内存监控脚本
├── monitor_long_stability.ps1    # 长时间稳定性监控
└── requirements_inference.txt    # Python 推理依赖
```

### 运行测试

#### Go 基准测试

```bash
# 运行 intra=1 测试
go run test/benchmark_go_std_intra1.go

# 运行 intra=2 测试
go run test/benchmark_go_std_intra2.go

# 运行 intra=4 测试
go run test/benchmark_go_std_intra4.go

# 运行长时间稳定性测试
go run test/benchmark_go_long_stability.go
```

#### Python 基准测试

```bash
# 安装依赖
pip install -r test/requirements_inference.txt

# 运行 Python 测试
python test/inference_align.py
```

### 内存监控

#### 监控 Go 内存

```bash
# 启动内存监控脚本
./test/monitor_go_memory.ps1

# 同时运行 Go 测试程序
go run test/benchmark_go_std_intra1.go
```

#### 监控 Python 内存

```bash
# 启动内存监控脚本
./test/monitor_python_memory.ps1

# 同时运行 Python 测试程序
python test/inference_align.py
```

### 测试结果

测试结果将存储在 `results/` 目录中，包含：

- 延迟指标：Avg、p50、p90、p99
- 内存使用：Peak RSS、Stable RSS
- 长时间稳定性：内存泄漏检测、推理稳定性

## 📊 性能对比

### 推理性能（YOLO11x）

| 实现语言 | Avg (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|---------|----------|----------|----------|----------|
| Python  | 3382.21  | 3711.85  | 4184.99  | 4590.05  |
| Go      | 1087.35  | 1088.36  | 1207.51  | 1260.42  |

### 内存使用（YOLO11x）

| 实现语言 | Peak RSS (MB) | Stable RSS (MB) |
|---------|---------------|----------------|
| Python  | 4280.00       | 3860.00        |
| Go      | 3650.00       | 3250.00        |
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

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📄 许可证

MIT License

## 🙏 致谢

- [ultralytics/yolov11](https://docs.ultralytics.com/models/yolo11/) - YOLOv11 模型
- [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) - Go语言ONNX Runtime绑定
- [Go编程语言](https://go.dev/) - Go语言开发
- 人工智能后面的所有人类, 感谢所有开源项目提供的帮助
