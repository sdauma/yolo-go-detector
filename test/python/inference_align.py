import onnxruntime as ort
import numpy as np
import cv2
import time

# ======================
# 配置区（与 Go 保持一致）
# ======================
MODEL_PATH = "./third_party/yolo11x.onnx"
IMAGE_PATH = "./assets/bus.jpg"

INPUT_W = 640
INPUT_H = 640
STRIDE = 32

CONF_THRES = 0.25
IOU_THRES = 0.7

# 明确指定 ONNX Runtime 版本兼容性
print(f"ONNX Runtime version: {ort.__version__}")

np.set_printoptions(suppress=True)

# ======================
# letterbox（严格可对齐，与 Go 版本保持一致）
# ======================
def letterbox(
        img,
        new_shape=(640, 640),
        stride=32,
        color=(114, 114, 114)
):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    # 计算缩放比例，与 Go 版本一致
    r = min(float(new_h) / h, float(new_w) / w)

    # 计算缩放后的尺寸
    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    # 计算填充量
    dw = new_w - resized_w
    dh = new_h - resized_h

    # 居中填充
    dw /= 2
    dh /= 2

    # 显式指定 resize 算法为线性插值，与 Go 版本一致
    img = cv2.resize(
        img,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    # 计算上下左右填充量，确保与 Go 版本一致
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # 显式指定填充位置和颜色
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img, r, left, top

# ======================
# 预处理（完全显式，与 Go 版本保持一致）
# ======================
def preprocess(image_path):
    # 1. 读取图像（BGR格式）
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"无法读取图像: {image_path}"

    # 2. 显式转换为 RGB 格式，与 Go 版本一致
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. 应用 letterbox 变换
    img_lb, scale, pad_x, pad_y = letterbox(
        img_rgb,
        (INPUT_H, INPUT_W),
        STRIDE
    )

    # 4. 显式转换为 float32 并归一化到 [0, 1]
    img = img_lb.astype(np.float32) / 255.0

    # 5. 显式转换为 CHW 格式
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW

    # 6. 显式添加 batch 维度，转换为 NCHW 格式
    img = np.expand_dims(img, axis=0)   # → NCHW

    # 7. 确保数据类型为 float32
    img = img.astype(np.float32)

    return img, scale, pad_x, pad_y, img_bgr.shape[:2]

# ======================
# ONNX Runtime Session（与 Go 版本保持一致）
# ======================
def create_session(model_path):
    # 显式创建会话选项
    so = ort.SessionOptions()
    
    # 显式设置图优化级别，与 Go 版本一致
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 显式设置线程数，与 Go 版本一致 (intra=4, inter=1)
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1

    # 显式指定执行提供者为 CPU，与 Go 版本一致
    providers = ["CPUExecutionProvider"]

    # 显式创建 ONNX Runtime 会话
    session = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=providers
    )
    return session

# ======================
# 推理（infer only，与 Go 版本保持一致）
# ======================
def infer(session, input_tensor):
    # 1. 获取输入张量名称，与 Go 版本一致
    input_name = session.get_inputs()[0].name

    # 2. 执行 ONNX Runtime 原生推理
    t0 = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    t1 = time.time()

    # 3. 打印推理时间
    print(f"Inference time: {(t1 - t0)*1000:.3f} ms")
    
    return outputs

# ======================
# 输出检查（对齐用，与 Go 版本保持一致）
# ======================
def dump_output(outputs, max_rows=20):
    # 1. 获取第一个输出并展平
    out = outputs[0].reshape(-1)
    print(f"Output0 length: {len(out)}")

    # 2. 打印前 20 个 float 值，统一保留 6 位小数
    print("===== 模型输出前20个值 =====")
    for i in range(min(max_rows, len(out))):
        print(f"[{i}] {out[i]:.6f}")
    print("=========================")

# ======================
# 主流程（与 Go 版本保持一致）
# ======================
def main():
    print("========================================")
    print("ONNX Runtime 原生推理测试")
    print("========================================")
    
    # 1. 打印所有使用的参数值，确保论文严谨性
    print(f"模型路径: {MODEL_PATH}")
    print(f"图像路径: {IMAGE_PATH}")
    print(f"输入尺寸: {INPUT_W}x{INPUT_H}")
    print(f"步长: {STRIDE}")
    print(f"置信度阈值: {CONF_THRES}")
    print(f"IoU 阈值: {IOU_THRES}")
    print(f"线程配置: intra=4, inter=1")
    print("========================================")

    # 2. 创建 ONNX Runtime 会话
    session = create_session(MODEL_PATH)

    # 3. 执行预处理
    input_tensor, scale, pad_x, pad_y, orig_hw = preprocess(IMAGE_PATH)

    # 4. 打印预处理参数
    print(f"缩放比例: {scale:.6f}")
    print(f"左填充: {pad_x}")
    print(f"上填充: {pad_y}")
    print(f"原始尺寸: {orig_hw}")
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输入张量数据类型: {input_tensor.dtype}")
    print("========================================")

    # 5. Warmup 阶段
    print("\nWarming up...")
    warmup_runs = 10
    for i in range(warmup_runs):
        outputs = infer(session, input_tensor)
        if i % 2 == 0:
            print(f"Warm-up {i+1}/{warmup_runs} done")

    # 6. Benchmark 阶段
    print("\nRunning benchmark...")
    benchmark_runs = 30
    times = []

    for i in range(benchmark_runs):
        if i % 5 == 0:
            print(f"Progress: {i}/{benchmark_runs}")
        
        t0 = time.time()
        outputs = infer(session, input_tensor)
        t1 = time.time()
        dt = (t1 - t0) * 1000.0  # 转换为毫秒
        times.append(dt)

    # 7. 计算性能指标
    if times:
        avg_time = sum(times) / len(times)
        times.sort()
        p50_time = times[len(times) // 2]
        p90_time = times[int(len(times) * 0.9)]
        p99_time = times[int(len(times) * 0.99)]
        min_time = min(times)
        max_time = max(times)

        print("\n===== 测试结果 =====")
        print(f"Python avg: {avg_time:.3f} ms")
        print(f"Python p50: {p50_time:.3f} ms")
        print(f"Python p90: {p90_time:.3f} ms")
        print(f"Python p99: {p99_time:.3f} ms")
        print(f"Python min: {min_time:.3f} ms")
        print(f"Python max: {max_time:.3f} ms")
        print(f"\nTotal runs: {len(times)}")

    # 8. 打印输出结果
    print("\n========================================")
    dump_output(outputs, max_rows=20)
    print("========================================")
    print("测试完成")
    print("========================================")

if __name__ == "__main__":
    main()
