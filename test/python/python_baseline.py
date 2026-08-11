# -*- coding: utf-8 -*-
# python_baseline.py
# Python Baseline Test - Baseline Execution Path
# 
# Important Declaration (P0 Principle):
# This test uses Python baseline Session API (InferenceSession).
# Python ONNX Runtime's run() method performs data copy on each call (no Go-style I/O Binding).
# Per P0 principle, this test is for observation only, not for language-level performance conclusions.
# 
# Technical Notes:
# - Fixed thread configuration: intra_op_num_threads=12, inter_op_num_threads=1
# - All SessionOptions parameters explicitly set (P2 principle)
#
# Test Purpose:
# - Measure Python baseline performance under fixed thread configuration
# - Provide reference baseline for comparison with Go side
# - Not for language-level performance conclusions

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# 鍥哄畾闅忔満绉嶅瓙锛岀‘淇濆彲澶嶇幇
np.random.seed(12345)

# 鑾峰彇褰撳墠宸ヤ綔鐩綍
current_dir = os.path.dirname(os.path.abspath(__file__))

# 鏋勫缓model璺緞
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))

# 鏋勫缓椤圭洰鏍硅矾寰?
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 妫€鏌odel鏂囦欢鏄惁瀛樺湪
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
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
    print("===== Python Baseline Test ====")
    
    # Create Session
    print("Creating InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # Explicitly set all SessionOptions parameters (P2 principle: no default value dependency)
        # Thread configuration - 12 threads, consistent with other tests
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        
        # Log configuration (disable all logs to avoid IO interference)
        sess_options.log_severity_level = 3
        
        # Profiling configuration (disable profiling to avoid overhead)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Memory pool configuration (enable memory pool reuse)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # All unspecified Session parameters use ONNX Runtime 1.23.2 official default values
        
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        print("InferenceSession created successfully!")
    except Exception as e:
        print(f"Error: Failed to create InferenceSession: {e}")
        sys.exit(1)

    # Get input information
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Use identical input data as Go (loaded from file with fixed seed)
    print("Loading input data...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"Input data loaded successfully: {input_data_path}")
    except Exception as e:
        print(f"Failed to load input data: {e}")
        sys.exit(1)

    # Memory sample point 1: After Session creation, before warmup (Start PM)
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().private / 1024 / 1024

    # Warmup
    print("Warming up...")
    for _ in range(20):
        sess.run(None, {input_name: input_data})

    # Memory sample point 2: After warmup
    warmup_rss = process.memory_info().private / 1024 / 1024

    # Benchmark
    print("Running benchmark...")
    runs = 200
    times = []
    peak_rss = start_rss

    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        times.append(dt)

        # Sample memory, record peak
        current_rss = process.memory_info().private / 1024 / 1024
        if current_rss > peak_rss:
            peak_rss = current_rss

    # Memory sample point 3: Stable value after benchmark
    stable_rss = process.memory_info().private / 1024 / 1024

    # Calculate results
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
    print("===== Python Baseline Test (10 runs) =====")

    # Run 10 tests
    num_runs = 10
    results = []

    for i in range(num_runs):
        print(f"\n===== Run {i+1} =====")
        result = run_benchmark()
        results.append(result)

        print(f"avg_latency: {result.avg_latency:.5f} ms")
        print(f"P50寤惰繜: {result.p50_latency:.5f} ms")
        print(f"P90寤惰繜: {result.p90_latency:.5f} ms")
        print(f"P99寤惰繜: {result.p99_latency:.5f} ms")
        print(f"min_latency: {result.min_latency:.5f} ms")
        print(f"max_latency: {result.max_latency:.5f} ms")
        print(f"Start PM: {result.start_rss:.5f} MB")
        print(f"Peak PM: {result.peak_rss:.5f} MB")
        print(f"Stable PM: {result.stable_rss:.5f} MB")
        print(f"PM Drift: {result.stable_rss - result.start_rss:.5f} MB")

    # 璁＄畻骞冲潎鍊?
    avg_latency = sum(r.avg_latency for r in results) / num_runs
    p50_latency = sum(r.p50_latency for r in results) / num_runs
    p90_latency = sum(r.p90_latency for r in results) / num_runs
    p99_latency = sum(r.p99_latency for r in results) / num_runs
    min_latency = sum(r.min_latency for r in results) / num_runs
    max_latency = sum(r.max_latency for r in results) / num_runs
    start_rss = sum(r.start_rss for r in results) / num_runs
    peak_rss = sum(r.peak_rss for r in results) / num_runs
    stable_rss = sum(r.stable_rss for r in results) / num_runs

    print(f"\n===== Average of 10 runs =====")
    print(f"Average latency: {avg_latency:.5f} ms")
    print(f"P50 latency: {p50_latency:.5f} ms")
    print(f"P90 latency: {p90_latency:.5f} ms")
    print(f"P99 latency: {p99_latency:.5f} ms")
    print(f"Min latency: {min_latency:.5f} ms")
    print(f"Max latency: {max_latency:.5f} ms")
    print(f"Start PM: {start_rss:.5f} MB")
    print(f"Peak PM: {peak_rss:.5f} MB")
    print(f"Stable PM: {stable_rss:.5f} MB")
    print(f"PM Drift: {stable_rss - start_rss:.5f} MB")

    # Save detailed log
    log_path = os.path.join(base_path, "results", "python_baseline_detailed_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results):
            f.write(f"===== 绗?{i+1} 娆℃祴璇?=====\n")
            f.write(f"avg_latency: {r.avg_latency:.5f} ms\n")
            f.write(f"P50寤惰繜: {r.p50_latency:.5f} ms\n")
            f.write(f"P90寤惰繜: {r.p90_latency:.5f} ms\n")
            f.write(f"P99寤惰繜: {r.p99_latency:.5f} ms\n")
            f.write(f"min_latency: {r.min_latency:.5f} ms\n")
            f.write(f"max_latency: {r.max_latency:.5f} ms\n")
            f.write(f"Start PM: {r.start_rss:.5f} MB\n")
            f.write(f"Peak PM: {r.peak_rss:.5f} MB\n")
            f.write(f"Stable PM: {r.stable_rss:.5f} MB\n")
            f.write(f"PM Drift: {r.stable_rss - r.start_rss:.5f} MB\n")
            f.write("\n")

        f.write("===== Average of 10 runs =====\n")
        f.write(f"avg_latency: {avg_latency:.5f} ms\n")
        f.write(f"P50寤惰繜: {p50_latency:.5f} ms\n")
        f.write(f"P90寤惰繜: {p90_latency:.5f} ms\n")
        f.write(f"P99寤惰繜: {p99_latency:.5f} ms\n")
        f.write(f"min_latency: {min_latency:.5f} ms\n")
        f.write(f"max_latency: {max_latency:.5f} ms\n")
        f.write(f"Start PM: {start_rss:.5f} MB\n")
        f.write(f"Peak PM: {peak_rss:.5f} MB\n")
        f.write(f"Stable PM: {stable_rss:.5f} MB\n")
        f.write(f"PM Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"\n璇︾粏鏃ュ織宸蹭繚瀛樺埌: {log_path}")

    # Save average results
    result_path = os.path.join(base_path, "results", "python_baseline_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Baseline Test Results (10 runs average) =====\n")
        f.write(f"avg_latency: {avg_latency:.5f} ms\n")
        f.write(f"P50 latency: {p50_latency:.5f} ms\n")
        f.write(f"P90 latency: {p90_latency:.5f} ms\n")
        f.write(f"P99 latency: {p99_latency:.5f} ms\n")
        f.write(f"min_latency: {min_latency:.5f} ms\n")
        f.write(f"max_latency: {max_latency:.5f} ms\n")
        f.write("\n===== Memory Usage (10 runs average) =====\n")
        f.write(f"Start PM: {start_rss:.5f} MB\n")
        f.write(f"Peak PM: {peak_rss:.5f} MB\n")
        f.write(f"Stable PM: {stable_rss:.5f} MB\n")
        f.write(f"PM Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"Results saved to: {result_path}")

    # 淇濆瓨鏈€鍚庝竴娆℃祴璇曠殑鍘熷寤惰繜鏁版嵁锛堢敤浜庣敓鎴愮绾垮浘锛?
    # Intermediate data keeps 5 decimal places, conforming to core journal standards
    latency_data_path = os.path.join(base_path, "results", "python_baseline_latency_data.txt")
    with open(latency_data_path, 'w', encoding='utf-8') as f:
        for t in results[num_runs-1].times:
            f.write(f"{t:.5f}\n")

    print(f"鍘熷寤惰繜鏁版嵁宸蹭繚瀛樺埌: {latency_data_path}")
    print("Test completed!")

if __name__ == "__main__":
    main()

