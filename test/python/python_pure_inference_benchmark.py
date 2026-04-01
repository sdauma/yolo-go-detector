# python_pure_inference_benchmark.py
# Python 纯推理测试 - 输入只加载一次，循环复用
#
# 测试目的：
# - 测量纯推理延迟（不包含IO开销）
# - 输入数据只加载一次，循环复用
# - 确保测试结果反映真实推理性能

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# 固定随机种子，确保可复现
np.random.seed(12345)

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

@dataclass
class BenchmarkResult:
    avg_latency: float
    std_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    min_latency: float
    max_latency: float
    start_rss: float
    peak_rss: float
    stable_rss: float
    times: list

def run_benchmark(model_name, model_path):
    print(f"===== Python 纯推理测试 - {model_name} ====")
    
    # 不绑定CPU核心，让系统自由调度（匹配Go的默认行为）
    process = psutil.Process(os.getpid())
    print("CPU核心调度：系统默认")
    
    # 创建 Session
    print("创建 InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # 显式设置所有 SessionOptions 参数
        # 线程配置 - 12线程，匹配Go的默认行为
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        
        # 日志配置（关闭所有日志，避免日志IO干扰性能）
        sess_options.log_severity_level = 3
        
        # 性能分析配置（关闭性能分析，避免额外开销）
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 内存池配置（启用内存池复用）
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 所有未提及的Session参数均使用ONNX Runtime 1.23.2官方默认值
        
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        print("InferenceSession 创建成功!")
    except Exception as e:
        print(f"错误: 创建 InferenceSession 失败: {e}")
        sys.exit(1)

    # 获取输入信息
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"输入形状: {input_shape}")

    # 加载输入数据（只加载一次，循环复用）
    print("加载输入数据...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"输入数据加载成功: {input_data_path}")
        print(f"输入数据形状: {input_data.shape}")
        print(f"输入数据类型: {input_data.dtype}")
    except Exception as e:
        print(f"加载输入数据失败: {e}")
        sys.exit(1)

    # 内存采样点 1：Session 创建后、warmup 前（Start RSS）
    start_rss = process.memory_info().rss / 1024 / 1024

    # Warmup
    print("Warming up...")
    for _ in range(20):  # 20次warmup
        sess.run(None, {input_name: input_data})

    # 内存采样点 2：Warmup 后
    warmup_rss = process.memory_info().rss / 1024 / 1024

    # Benchmark - 纯推理，输入复用
    print("Running pure inference benchmark...")
    runs = 2000  # 2000次推理
    times = []
    peak_rss = start_rss

    for i in range(runs):
        t0 = time.perf_counter()
        # 每次推理都创建新的input tensor副本，避免CPU cache效应
        input_tensor = input_data.copy()
        sess.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        times.append(dt)

        # 采样内存，记录峰值
        current_rss = process.memory_info().rss / 1024 / 1024
        if current_rss > peak_rss:
            peak_rss = current_rss

    # 内存采样点 3：Benchmark 后稳定值
    stable_rss = process.memory_info().rss / 1024 / 1024

    # 计算结果
    avg_latency = sum(times) / len(times)
    std_latency = np.std(times)
    min_latency = min(times)
    max_latency = max(times)
    p50_latency = np.percentile(times, 50)
    p90_latency = np.percentile(times, 90)
    p95_latency = np.percentile(times, 95)
    p99_latency = np.percentile(times, 99)  # 额外计算p99

    return BenchmarkResult(
        avg_latency=avg_latency,
        std_latency=std_latency,
        p50_latency=p50_latency,
        p90_latency=p90_latency,
        p95_latency=p95_latency,
        min_latency=min_latency,
        max_latency=max_latency,
        start_rss=start_rss,
        peak_rss=peak_rss,
        stable_rss=stable_rss,
        times=times
    )

def main():
    print("===== Python 纯推理测试 ====")
    print("测试配置：")
    print("- 线程数: 12")
    print("- 输入数据: 只加载一次，循环复用")
    print("- 推理次数: 2000次")
    print("- Warmup: 20次")
    print()

    # 测试 YOLO11x 模型
    print("\n===== 测试 YOLO11x 模型 =====")
    yolo11x_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
    if not os.path.exists(yolo11x_path):
        print(f"错误: YOLO11x模型文件不存在: {yolo11x_path}")
        sys.exit(1)
    
    yolo11x_result = run_benchmark("YOLO11x", yolo11x_path)
    
    print(f"\nYOLO11x 测试结果:")
    print(f"平均延迟: {yolo11x_result.avg_latency:.3f} ms")
    print(f"标准差: {yolo11x_result.std_latency:.3f} ms")
    print(f"P50延迟: {yolo11x_result.p50_latency:.3f} ms")
    print(f"P90延迟: {yolo11x_result.p90_latency:.3f} ms")
    print(f"P95延迟: {yolo11x_result.p95_latency:.3f} ms")
    print(f"最小延迟: {yolo11x_result.min_latency:.3f} ms")
    print(f"最大延迟: {yolo11x_result.max_latency:.3f} ms")
    print(f"Start RSS: {yolo11x_result.start_rss:.2f} MB")
    print(f"Peak RSS: {yolo11x_result.peak_rss:.2f} MB")
    print(f"Stable RSS: {yolo11x_result.stable_rss:.2f} MB")
    print(f"RSS Drift: {yolo11x_result.stable_rss - yolo11x_result.start_rss:.2f} MB")

    # 测试 YOLO11n 模型
    print("\n===== 测试 YOLO11n 模型 =====")
    yolo11n_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))
    if not os.path.exists(yolo11n_path):
        print(f"错误: YOLO11n模型文件不存在: {yolo11n_path}")
        sys.exit(1)
    
    yolo11n_result = run_benchmark("YOLO11n", yolo11n_path)
    
    print(f"\nYOLO11n 测试结果:")
    print(f"平均延迟: {yolo11n_result.avg_latency:.3f} ms")
    print(f"标准差: {yolo11n_result.std_latency:.3f} ms")
    print(f"P50延迟: {yolo11n_result.p50_latency:.3f} ms")
    print(f"P90延迟: {yolo11n_result.p90_latency:.3f} ms")
    print(f"P95延迟: {yolo11n_result.p95_latency:.3f} ms")
    print(f"最小延迟: {yolo11n_result.min_latency:.3f} ms")
    print(f"最大延迟: {yolo11n_result.max_latency:.3f} ms")
    print(f"Start RSS: {yolo11n_result.start_rss:.2f} MB")
    print(f"Peak RSS: {yolo11n_result.peak_rss:.2f} MB")
    print(f"Stable RSS: {yolo11n_result.stable_rss:.2f} MB")
    print(f"RSS Drift: {yolo11n_result.stable_rss - yolo11n_result.start_rss:.2f} MB")

    # 获取系统信息
    import platform
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_model = platform.processor()
    system_info = platform.platform()
    
    # 保存结果
    result_path = os.path.join(base_path, "results", "python_pure_inference_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python 纯推理测试结果 =====\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"系统信息: {system_info}\n")
        f.write(f"CPU型号: {cpu_model}\n")
        f.write("测试配置：\n")
        f.write("- 线程数: 12\n")
        f.write("- 输入数据: 只加载一次，每次推理创建副本\n")
        f.write("- 推理次数: 2000次\n")
        f.write("- Warmup: 20次\n")
        f.write("\n")
        
        f.write("===== YOLO11x 测试结果 =====\n")
        f.write(f"模型: YOLO11x\n")
        f.write(f"平均延迟: {yolo11x_result.avg_latency:.5f} ms\n")
        f.write(f"标准差: {yolo11x_result.std_latency:.5f} ms\n")
        f.write(f"P50延迟: {yolo11x_result.p50_latency:.5f} ms\n")
        f.write(f"P90延迟: {yolo11x_result.p90_latency:.5f} ms\n")
        f.write(f"P95延迟: {yolo11x_result.p95_latency:.5f} ms\n")
        f.write(f"最小延迟: {yolo11x_result.min_latency:.5f} ms\n")
        f.write(f"最大延迟: {yolo11x_result.max_latency:.5f} ms\n")
        f.write(f"Start RSS: {yolo11x_result.start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {yolo11x_result.peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {yolo11x_result.stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {yolo11x_result.stable_rss - yolo11x_result.start_rss:.5f} MB\n")
        f.write("\n")
        
        f.write("===== YOLO11n 测试结果 =====\n")
        f.write(f"模型: YOLO11n\n")
        f.write(f"平均延迟: {yolo11n_result.avg_latency:.5f} ms\n")
        f.write(f"标准差: {yolo11n_result.std_latency:.5f} ms\n")
        f.write(f"P50延迟: {yolo11n_result.p50_latency:.5f} ms\n")
        f.write(f"P90延迟: {yolo11n_result.p90_latency:.5f} ms\n")
        f.write(f"P95延迟: {yolo11n_result.p95_latency:.5f} ms\n")
        f.write(f"最小延迟: {yolo11n_result.min_latency:.5f} ms\n")
        f.write(f"最大延迟: {yolo11n_result.max_latency:.5f} ms\n")
        f.write(f"Start RSS: {yolo11n_result.start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {yolo11n_result.peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {yolo11n_result.stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {yolo11n_result.stable_rss - yolo11n_result.start_rss:.5f} MB\n")

    print(f"\n结果已保存到: {result_path}")
    print("测试完成!")

if __name__ == "__main__":
    main()
