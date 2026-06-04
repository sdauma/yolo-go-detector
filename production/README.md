# YOLO-Go 生产级摄像头检测系统

基于论文 **"YOLO-Go: Session Pool 架构"** 的生产级部署代码。全集团共注册 **2809 路**摄像头，每4小时动态刷新在线列表，平时在线约 **2600 路**。

## 架构概述

```
config.yaml (摄像头配置)
       │
       ▼
  Scheduler (每5秒一轮)
       │
       ├──→ CameraManager.GetOnlineCameras()  ← 每4小时刷新在线列表
       │
       ├──→ HTTP 并发取图 (goroutine 池, 限流100并发)
       │         │
       │         ▼  图片 []byte (纯内存, 不落盘)
       │
       ├──→ DetectionPipeline (Session Pool + YOLO 推理)
       │         │
       │         ▼  engine.SessionPool + engine.Postprocessor
       │
       └──→ OutputWriter
                ├─ JSONL 写入 (每轮一行, ~260KB)
                └─ 告警图片 (仅检测到目标时存盘)
```

## 快速开始

### 前置条件

- Go 1.25+
- ONNX Runtime 动态库（放在 `../third_party/` 目录）
- YOLO11x ONNX 模型（放在 `../third_party/yolo11x.onnx`）

### 配置

复制并编辑配置文件：

```bash
cp config.yaml config.prod.yaml
# 编辑 config.prod.yaml 填入真实信息
```

关键配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `video_api.base_url` | 视联网 API 地址 | `https://172.18.137.20:9502` |
| `video_api.username` | 登录用户名 | (必填) |
| `video_api.password` | 登录密码 | (必填) |
| `video_api.refresh_interval_minutes` | 在线列表刷新间隔 | `240` (4小时) |
| `detection.model_path` | ONNX 模型路径 | `../third_party/yolo11x.onnx` |
| `detection.pool_size` | Session Pool 大小 | `0` (自动=CPU核数) |
| `detection.intra_op_threads` | 每Session线程数 | `0` (自动=CPU/Pool，推荐) |
| `scheduler.round_interval_seconds` | 每轮检测间隔 | `5` |
| `scheduler.fetch_concurrency` | 并发取图数 | `100` |
| `output.jsonl_dir` | JSONL 输出目录 | `./output` |
| `output.alert_image_dir` | 告警图片目录 | `./output/alerts` |

### 编译运行

```bash
go build -o yolo-detector .
./yolo-detector -config ./config.prod.yaml
```

### 结果输出

#### JSONL 格式（每行一条）

```json
{"channel_code":"100005131004","channel_name":"1号泵房","timestamp":"2026-06-03 10:05:30","width":1920,"height":1080,"detections":[{"class":"person","confidence":0.85,"xmin":100.5,"ymin":200.3,"xmax":300.1,"ymax":500.2}],"fetch_ms":234,"infer_ms":45}
```

#### 每轮摘要

```json
{"type":"round_summary","timestamp":"2026-06-03 10:05:30","total_online":2600,"success_count":2598,"fail_count":2,"alert_count":3,"round_total_ms":4823,"avg_fetch_ms":234,"avg_infer_ms":45,"pool_active":14,"pool_idle":0}
```

## 设计原则

### 全程内存处理

- HTTP 响应体 → `image.Decode()` → 张量 → 推理 → 结果输出，不写一次盘
- 每轮检测约 2600 路在线摄像头，不存盘，JSONL 每轮仅 ~260KB

### 告警选择性落盘

仅当检测到配置的 `alert_classes`（人员/车辆等）中目标时，才将原图+标注图存盘。

### 不在线摄像头处理

- 每4小时刷新一次在线列表，只检测在线摄像头
- 离线摄像头不在本轮检测范围内，避免重复检测同一张旧图
- 不做 MD5 图片去重或 Redis 缓存——这是工程折中，4小时刷新间隔足以满足生产需求

## 代码结构

```
production/
├── main.go               # 入口，组装模块
├── config.go              # 配置加载
├── config.yaml            # 示例配置
├── camera.go              # 摄像头管理（在线列表/Token/HTTP取图）
├── detector.go            # YOLO 检测流水线（预处理+推理+后处理）
├── scheduler.go           # 轮巡调度器
├── output.go              # JSONL 输出 + 告警图片保存
├── go.mod / go.sum        # Go 模块
└── README.md              # 本文档
```

## 与论文的关系

本代码是论文 **"YOLO-Go: Session Pool 架构"** 第 5 节工程案例的实际部署代码：

- 复用论文核心成果 `engine.SessionPool` 进行 Session 池化管理
- 复用 `engine.Postprocessor` 进行 YOLO 后处理和 NMS
- 验证论文提出的全集团 2809 路摄像头并发检测在实际生产系统中的可行性（在线数动态变化，约 2600 路）

## 性能优化记录

2026-06-04 通过 `test/batch_verify/` 批量验证性能调优，发现并修复了两个影响生产环境的并发 Bug：

| Bug | 影响 | 修复 |
|-----|------|------|
| **池泄漏**（`GetSession()` 非阻塞） | Session 数超容量 → CPU 风暴 | 改为阻塞等待 + 预建满池 Session |
| **CPU 过度订阅**（未设线程限制） | 每 Session 默认 12 线程 → 144+ 线程争抢 | 添加 `SetIntraOpNumThreads`，自动计算 |

修复后 `engine.SessionPool` 在生产配置（pool=12, intra_op=1）下可稳定运行，2600 路摄像头每路推理延迟可控。配置新增 `detection.intra_op_threads` 参数（0 = 自动计算 CPU/PoolSize，推荐）。

## 许可证

MIT License
