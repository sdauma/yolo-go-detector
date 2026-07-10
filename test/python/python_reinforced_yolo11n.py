# -*- coding: utf-8 -*-
# python_reinforced_yolo11n.py
# Python YOLO11n small model reinforced test - 10 rounds x 200 inferences
#
# Technical:
# - Uses Python baseline Session interface (InferenceSession)
# - Explicitly configures thread params via SessionOptions (intraOp=12, interOp=1)
# - Uses sess.run() standard call path, no I/O Binding
# - Does not bind CPU cores, lets system freely schedule
#
# Purpose:
# - Compare difference patterns between YOLO11n and YOLO11x
# - Verify runtime overhead ratio changes

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# Fixed random seed for reproducibility
np.random.seed(12345)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build model path - YOLO11n
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# Build project root path
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Check model file
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
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
    print("===== Python YOLO11n Reinforced Test ====")

    # Don't bind CPU cores, let system freely schedule (matches Go default behavior)
    process = psutil.Process(os.getpid())
    print("CPU core scheduling: system default")

    # Create Session
    print("Creating InferenceSession...")
    try:
        sess_options = ort.SessionOptions()

        # Explicitly set all SessionOptions params (P2 principle: no reliance on defaults)
        # Thread config
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1

        # Log config (disable all logs to avoid log IO interference)
        sess_options.log_severity_level = 3

        # Execution mode config (sequential to avoid extra overhead)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Memory pool config (enable memory pool reuse)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # All unmentioned Session params use ONNX Runtime 1.23.2 official defaults
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        print("InferenceSession created successfully!")
    except Exception as e:
        print(f"Error: Failed to create InferenceSession: {e}")
        sys.exit(1)

    # Get input info
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"Input shape: {input_shape}")

    # Use input data consistent with Go (loaded from file, fixed seed)
    print("Loading input data...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"Input data loaded: {input_data_path}")
    except Exception as e:
        print(f"Failed to load input data: {e}")
        sys.exit(1)

    # Memory sample point 1: after Session creation, before Warmup (Start RSS)
    start_rss = process.memory_info().private / 1024 / 1024

    # Warmup
    print("Warming up...")
    for _ in range(20):  # 20 warmup inferences
        sess.run(None, {input_name: input_data})

    # Memory sample point 2: after Warmup
    warmup_rss = process.memory_info().private / 1024 / 1024

    # Benchmark
    print("Running benchmark...")
    runs = 200  # 200 inferences per round
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

    # Memory sample point 3: stable value after Benchmark
    stable_rss = process.memory_info().private / 1024 / 1024

    # Compute results
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
    print("===== Python YOLO11n Reinforced Test (10 rounds) =====")

    # Run 10 tests
    num_runs = 10
    results = []

    for i in range(num_runs):
        print(f"\n===== Round {i+1} =====")
        result = run_benchmark()
        results.append(result)

        print(f"Avg latency: {result.avg_latency:.3f} ms")
        print(f"Std dev: {result.std_latency:.3f} ms")
        print(f"P50 latency: {result.p50_latency:.3f} ms")
        print(f"P90 latency: {result.p90_latency:.3f} ms")
        print(f"P95 latency: {result.p95_latency:.3f} ms")
        print(f"Min latency: {result.min_latency:.3f} ms")
        print(f"Max latency: {result.max_latency:.3f} ms")
        print(f"Start RSS: {result.start_rss:.2f} MB")
        print(f"Peak RSS: {result.peak_rss:.2f} MB")
        print(f"Stable RSS: {result.stable_rss:.2f} MB")
        print(f"RSS Drift: {result.stable_rss - result.start_rss:.2f} MB")

    # Compute averages
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

    # Compute throughput (using actual wall-clock time)
    inferences_per_run = 200  # 200 inferences per round
    total_inferences = num_runs * inferences_per_run  # 10 rounds x 200 = 2000
    total_time_seconds = sum(r.avg_latency * inferences_per_run for r in results) / 1000.0
    throughput = total_inferences / total_time_seconds

    print(f"\n===== 10-Round Average =====")
    print(f"Avg latency: {avg_latency:.3f} ms")
    print(f"Std dev: {std_latency:.3f} ms")
    print(f"P50 latency: {p50_latency:.3f} ms")
    print(f"P90 latency: {p90_latency:.3f} ms")
    print(f"P95 latency: {p95_latency:.3f} ms")
    print(f"Min latency: {min_latency:.3f} ms")
    print(f"Max latency: {max_latency:.3f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Start RSS: {start_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    print(f"Stable RSS: {stable_rss:.2f} MB")
    print(f"RSS Drift: {stable_rss - start_rss:.2f} MB")

    # Get system info
    import platform
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_model = platform.processor()
    system_info = platform.platform()

    # Save detailed log
    log_path = os.path.join(base_path, "results", "python_yolo11n_reinforced_12threads_detailed_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("===== Python YOLO11n Reinforced Test Results =====\n")
        f.write(f"Test time: {timestamp}\n")
        f.write(f"System info: {system_info}\n")
        f.write(f"CPU model: {cpu_model}\n")
        f.write("Test config:\n")
        f.write("- Threads: 12\n")
        f.write("- Rounds: 10\n")
        f.write("- Inferences per round: 200\n")
        f.write("- Warmup: 20\n")
        f.write("\n")

        for i, r in enumerate(results):
            f.write(f"===== Round {i+1} =====\n")
            f.write(f"Avg latency: {r.avg_latency:.5f} ms\n")
            f.write(f"Std dev: {r.std_latency:.5f} ms\n")
            f.write(f"P50 latency: {r.p50_latency:.5f} ms\n")
            f.write(f"P90 latency: {r.p90_latency:.5f} ms\n")
            f.write(f"P95 latency: {r.p95_latency:.5f} ms\n")
            f.write(f"Min latency: {r.min_latency:.5f} ms\n")
            f.write(f"Max latency: {r.max_latency:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n")
            f.write(f"Stable RSS: {r.stable_rss:.5f} MB\n")
            f.write(f"RSS Drift: {r.stable_rss - r.start_rss:.5f} MB\n")
            f.write("\n")

        f.write("===== 10-Round Average =====\n")
        f.write(f"Avg latency: {avg_latency:.5f} ms\n")
        f.write(f"Std dev: {std_latency:.5f} ms\n")
        f.write(f"P50 latency: {p50_latency:.5f} ms\n")
        f.write(f"P90 latency: {p90_latency:.5f} ms\n")
        f.write(f"P95 latency: {p95_latency:.5f} ms\n")
        f.write(f"Min latency: {min_latency:.5f} ms\n")
        f.write(f"Max latency: {max_latency:.5f} ms\n")
        f.write(f"Start RSS: {start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"\nDetailed log saved to: {log_path}")

    # Save average results
    result_path = os.path.join(base_path, "results", "python_yolo11n_reinforced_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python YOLO11n Reinforced Test Results (10 rounds) =====\n")
        for i, r in enumerate(results):
            f.write(f"Round {i+1} avg latency: {r.avg_latency:.5f} ms\n")
        f.write("\n===== 10-Round Average =====\n")
        f.write(f"Avg latency: {avg_latency:.5f} ms\n")
        f.write(f"Std dev: {std_latency:.5f} ms\n")
        f.write(f"P50 latency: {p50_latency:.5f} ms\n")
        f.write(f"P90 latency: {p90_latency:.5f} ms\n")
        f.write(f"P95 latency: {p95_latency:.5f} ms\n")
        f.write(f"Min latency: {min_latency:.5f} ms\n")
        f.write(f"Max latency: {max_latency:.5f} ms\n")
        f.write(f"Start RSS: {start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {stable_rss - start_rss:.5f} MB\n")

    print(f"Results saved to: {result_path}")

if __name__ == "__main__":
    main()
