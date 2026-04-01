# python_reinforced_yolo11n.py
# Python YOLO11n轻模型强化测试 - 10轮×200次推理
# 
# 测试目的：
# - 对比YOLO11n与YOLO11x的差异模式
# - 验证runtime开销占比变化

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

# 构建模型路径 - YOLO11n
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

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

def run_benchmark():
    print("===== Python YOLO11n 强化测试 ====")
    
    # 不绑定CPU核心，让系统自由调度（匹配Go的默认行为）
    process = psutil.Process(os.getpid())
    print("CPU核心调度：系统默认")
    
    # 创建 Session
    print("创建 InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # 显式设置所有 SessionOptions 参数（P2原则：禁止依赖默认值）
        # 线程配置
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

    # 使用与 Go 完全一致的输入数据（从文件加载，使用固定种子）
    print("加载输入数据...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"输入数据加载成功: {input_data_path}")
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

    # Benchmark
    print("Running benchmark...")
    runs = 200  # 每轮200次推理
    times = []
    peak_rss = start_rss

    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
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
    print("===== Python YOLO11n 强化测试（10轮运行）=====")

    # 运行10次测试
    num_runs = 10
    results = []

    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 轮测试 =====")
        result = run_benchmark()
        results.append(result)

        print(f"平均延迟: {result.avg_latency:.3f} ms")
        print(f"标准差: {result.std_latency:.3f} ms")
        print(f"P50延迟: {result.p50_latency:.3f} ms")
        print(f"P90延迟: {result.p90_latency:.3f} ms")
        print(f"P95延迟: {result.p95_latency:.3f} ms")
        print(f"最小延迟: {result.min_latency:.3f} ms")
        print(f"最大延迟: {result.max_latency:.3f} ms")
        print(f"Start RSS: {result.start_rss:.2f} MB")
        print(f"Peak RSS: {result.peak_rss:.2f} MB")
        print(f"Stable RSS: {result.stable_rss:.2f} MB")
        print(f"RSS Drift: {result.stable_rss - result.start_rss:.2f} MB")

    # 计算平均值
    avg_latency = sum(r.avg_latency for r in results) / num_runs
    std_latency = sum(r.std_latency for r in results) / num_runs
    p50_latency = sum(r.p50_latency for r in results) / num_runs
    p90_latency = sum(r.p90_latency for r in results) / num_runs
    p95_latency = sum(r.p95_latency for r in results) / num_runs
    min_latency = sum(r.min_latency for r in results) / num_runs
    max_latency = sum(r.max_latency for r in results) / num_runs
    start_rss = sum(r.start_rss for r in results) / num_runs
    peak_rss = sum(r.peak_rss for r in results) / num_runs
    stable_rss = sum(r.stable_rss for r in results) / num_runs

    # 计算吞吐量
    inferences_per_run = 200  # 每轮200次推理
    total_inferences = num_runs * inferences_per_run  # 10轮 × 200次 = 2000次
    total_time_seconds = sum(r.avg_latency * inferences_per_run for r in results) / 1000.0
    throughput = total_inferences / total_time_seconds

    print(f"\n===== 10轮测试平均值 =====")
    print(f"平均延迟: {avg_latency:.3f} ms")
    print(f"标准差: {std_latency:.3f} ms")
    print(f"P50延迟: {p50_latency:.3f} ms")
    print(f"P90延迟: {p90_latency:.3f} ms")
    print(f"P95延迟: {p95_latency:.3f} ms")
    print(f"最小延迟: {min_latency:.3f} ms")
    print(f"最大延迟: {max_latency:.3f} ms")
    print(f"吞吐量: {throughput:.2f} images/sec")
    print(f"Start RSS: {start_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    print(f"Stable RSS: {stable_rss:.2f} MB")
    print(f"RSS Drift: {stable_rss - start_rss:.2f} MB")

    # 获取系统信息
    import platform
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_model = platform.processor()
    system_info = platform.platform()
    
    # 保存详细日志
    log_path = os.path.join(base_path, "results", "python_yolo11n_reinforced_12threads_detailed_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("===== Python YOLO11n 强化测试结果 =====\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"系统信息: {system_info}\n")
        f.write(f"CPU型号: {cpu_model}\n")
        f.write("测试配置：\n")
        f.write("- 线程数: 12\n")
        f.write("- 运行轮数: 10轮\n")
        f.write("- 每轮推理次数: 200次\n")
        f.write("- Warmup: 20次\n")
        f.write("\n")
        
        for i, r in enumerate(results):
            f.write(f"===== 第 {i+1} 轮测试 =====\n")
            f.write(f"平均延迟: {r.avg_latency:.5f} ms\n")
            f.write(f"标准差: {r.std_latency:.5f} ms\n")
            f.write(f"P50延迟: {r.p50_latency:.5f} ms\n")
            f.write(f"P90延迟: {r.p90_latency:.5f} ms\n")
            f.write(f"P95延迟: {r.p95_latency:.5f} ms\n")
            f.write(f"最小延迟: {r.min_latency:.5f} ms\n")
            f.write(f"最大延迟: {r.max_latency:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n")
            f.write(f"Stable RSS: {r.stable_rss:.5f} MB\n")
            f.write(f"RSS Drift: {r.stable_rss - r.start_rss:.5f} MB\n")
            f.write("\n")

        f.write("===== 10轮测试平均值 =====\n")
        f.write(f"平均延迟: {avg_latency:.5f} ms\n")
        f.write(f"标准差: {std_latency:.5f} ms\n")
        f.write(f"P50延迟: {p50_latency:.5f} ms\n")
        f.write(f"P90延迟: {p90_latency:.5f} ms\n")
        f.write(f"P95延迟: {p95_latency:.5f} ms\n")
        f.write(f"最小延迟: {min_latency:.5f} ms\n")
        f.write(f"最大延迟: {max_latency:.5f} ms\n")
        f.write(f"Start RSS: {start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"\n详细日志已保存到: {log_path}")

    # 保存平均值结果
    result_path = os.path.join(base_path, "results", "python_yolo11n_reinforced_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python YOLO11n 强化测试结果（10轮运行） =====\n")
        for i, r in enumerate(results):
            f.write(f"第{i+1}轮平均延迟: {r.avg_latency:.5f} ms\n")
        f.write("\n===== 10轮测试平均值 =====\n")
        f.write(f"平均延迟: {avg_latency:.5f} ms\n")
        f.write(f"标准差: {std_latency:.5f} ms\n")
        f.write(f"P50延迟: {p50_latency:.5f} ms\n")
        f.write(f"P90延迟: {p90_latency:.5f} ms\n")
        f.write(f"P95延迟: {p95_latency:.5f} ms\n")
        f.write(f"最小延迟: {min_latency:.5f} ms\n")
        f.write(f"最大延迟: {max_latency:.5f} ms\n")
        f.write(f"Start RSS: {start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"结果已保存到: {result_path}")

if __name__ == "__main__":
    main()
