# python_baseline.py
# Python 基准测试 - Baseline 执行路径
# 
# 重要声明（P0原则）：
# 本测试使用 Python baseline Session 接口（InferenceSession），不启用 I/O Binding。
# 根据 P0 原则，本测试仅用于观察现象，不用于语言级性能结论。
# 
# 测试目的：
# - 观察不同线程配置下的性能趋势
# - 验证 ONNX Runtime 的线程扩展性
# - 不用于语言级线程扩展性结论

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

# 构建模型路径
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    start_rss: float
    peak_rss: float
    stable_rss: float
    times: list

def run_benchmark():
    print("===== Python 基准测试 ====")
    
    # 创建 Session
    print("创建 InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # 显式设置所有 SessionOptions 参数（P2原则：禁止依赖默认值）
        # 线程配置
        sess_options.intra_op_num_threads = 4
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
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / 1024 / 1024

    # Warmup
    print("Warming up...")
    for _ in range(10):
        sess.run(None, {input_name: input_data})

    # 内存采样点 2：Warmup 后
    warmup_rss = process.memory_info().rss / 1024 / 1024

    # Benchmark
    print("Running benchmark...")
    runs = 100
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
    min_latency = min(times)
    max_latency = max(times)
    p50_latency = np.percentile(times, 50)
    p90_latency = np.percentile(times, 90)
    p99_latency = np.percentile(times, 99)

    return BenchmarkResult(
        avg_latency=avg_latency,
        p50_latency=p50_latency,
        p90_latency=p90_latency,
        p99_latency=p99_latency,
        min_latency=min_latency,
        max_latency=max_latency,
        start_rss=start_rss,
        peak_rss=peak_rss,
        stable_rss=stable_rss,
        times=times
    )

def main():
    print("===== Python 基准测试（5次运行） =====")

    # 运行5次测试
    num_runs = 5
    results = []

    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 次测试 =====")
        result = run_benchmark()
        results.append(result)

        print(f"平均延迟: {result.avg_latency:.3f} ms")
        print(f"P50延迟: {result.p50_latency:.3f} ms")
        print(f"P90延迟: {result.p90_latency:.3f} ms")
        print(f"P99延迟: {result.p99_latency:.3f} ms")
        print(f"最小延迟: {result.min_latency:.3f} ms")
        print(f"最大延迟: {result.max_latency:.3f} ms")
        print(f"Start RSS: {result.start_rss:.2f} MB")
        print(f"Peak RSS: {result.peak_rss:.2f} MB")
        print(f"Stable RSS: {result.stable_rss:.2f} MB")
        print(f"RSS Drift: {result.stable_rss - result.start_rss:.2f} MB")

    # 计算平均值
    avg_latency = sum(r.avg_latency for r in results) / num_runs
    p50_latency = sum(r.p50_latency for r in results) / num_runs
    p90_latency = sum(r.p90_latency for r in results) / num_runs
    p99_latency = sum(r.p99_latency for r in results) / num_runs
    min_latency = sum(r.min_latency for r in results) / num_runs
    max_latency = sum(r.max_latency for r in results) / num_runs
    start_rss = sum(r.start_rss for r in results) / num_runs
    peak_rss = sum(r.peak_rss for r in results) / num_runs
    stable_rss = sum(r.stable_rss for r in results) / num_runs

    print(f"\n===== 5次测试平均值 =====")
    print(f"平均延迟: {avg_latency:.3f} ms")
    print(f"P50延迟: {p50_latency:.3f} ms")
    print(f"P90延迟: {p90_latency:.3f} ms")
    print(f"P99延迟: {p99_latency:.3f} ms")
    print(f"最小延迟: {min_latency:.3f} ms")
    print(f"最大延迟: {max_latency:.3f} ms")
    print(f"Start RSS: {start_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    print(f"Stable RSS: {stable_rss:.2f} MB")
    print(f"RSS Drift: {stable_rss - start_rss:.2f} MB")

    # 保存详细日志
    log_path = os.path.join(base_path, "results", "python_baseline_detailed_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results):
            f.write(f"===== 第 {i+1} 次测试 =====\n")
            f.write(f"平均延迟: {r.avg_latency:.3f} ms\n")
            f.write(f"P50延迟: {r.p50_latency:.3f} ms\n")
            f.write(f"P90延迟: {r.p90_latency:.3f} ms\n")
            f.write(f"P99延迟: {r.p99_latency:.3f} ms\n")
            f.write(f"最小延迟: {r.min_latency:.3f} ms\n")
            f.write(f"最大延迟: {r.max_latency:.3f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.2f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.2f} MB\n")
            f.write(f"Stable RSS: {r.stable_rss:.2f} MB\n")
            f.write(f"RSS Drift: {r.stable_rss - r.start_rss:.2f} MB\n")
            f.write("\n")

        f.write("===== 5次测试平均值 =====\n")
        f.write(f"平均延迟: {avg_latency:.3f} ms\n")
        f.write(f"P50延迟: {p50_latency:.3f} ms\n")
        f.write(f"P90延迟: {p90_latency:.3f} ms\n")
        f.write(f"P99延迟: {p99_latency:.3f} ms\n")
        f.write(f"最小延迟: {min_latency:.3f} ms\n")
        f.write(f"最大延迟: {max_latency:.3f} ms\n")
        f.write(f"Start RSS: {start_rss:.2f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.2f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.2f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.2f} MB\n")

    print(f"\n详细日志已保存到: {log_path}")

    # 保存平均值结果
    result_path = os.path.join(base_path, "results", "python_baseline_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python 基准测试结果（5次运行平均值） =====\n")
        f.write(f"平均延迟: {avg_latency:.3f} ms\n")
        f.write(f"P50延迟: {p50_latency:.3f} ms\n")
        f.write(f"P90延迟: {p90_latency:.3f} ms\n")
        f.write(f"P99延迟: {p99_latency:.3f} ms\n")
        f.write(f"最小延迟: {min_latency:.3f} ms\n")
        f.write(f"最大延迟: {max_latency:.3f} ms\n")
        f.write("\n===== 内存使用情况（5次运行平均值） =====\n")
        f.write(f"Start RSS: {start_rss:.2f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.2f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.2f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.2f} MB\n")

    print(f"结果已保存到: {result_path}")

    # 保存最后一次测试的原始延迟数据（用于生成箱线图）
    latency_data_path = os.path.join(base_path, "results", "python_baseline_latency_data.txt")
    with open(latency_data_path, 'w', encoding='utf-8') as f:
        for t in results[num_runs-1].times:
            f.write(f"{t:.3f}\n")

    print(f"原始延迟数据已保存到: {latency_data_path}")
    print("测试完成!")

if __name__ == "__main__":
    main()
