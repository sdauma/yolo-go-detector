# -*- coding: utf-8 -*-
# python_pure_inference_benchmark.py
# Python pure inference benchmark - input loaded once, reused in loop
#
# Technical notes:
# - Uses Python baseline Session API (InferenceSession)
# - Explicitly configures thread params via SessionOptions (intraOp=12, interOp=1)
# - Uses sess.run() standard call path, no I/O Binding
# - Each inference creates input_data.copy() to avoid CPU cache effects
# - 2000 inferences, 20 warmup runs
#
# Test purpose:
# - Measure pure inference latency (excluding I/O overhead)
# - Input data loaded once, reused in loop
# - Ensure test results reflect real inference performance

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# Fixed random seed for reproducibility
np.random.seed(12345)

# Get current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build project root path
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
    print(f"===== Python Pure Inference Benchmark - {model_name} ====")
    
    # Do not bind CPU cores, let system schedule freely (matches Go default behavior)
    process = psutil.Process(os.getpid())
    print("CPU core scheduling: system default")
    
    # Create Session
    print("Creating InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # Explicitly set all SessionOptions params
        # Thread config - 12 threads, matching Go default behavior
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        
        # Log config (disable all logs to avoid log I/O interfering with performance)
        sess_options.log_severity_level = 3
        
        # Execution config (disable profiling to avoid extra overhead)
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

    # Load input data (load once, reuse in loop)
    print("Loading input data...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"Input data loaded successfully: {input_data_path}")
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data dtype: {input_data.dtype}")
    except Exception as e:
        print(f"Failed to load input data: {e}")
        sys.exit(1)

    # Memory sample point 1: after Session creation, before warmup (Start RSS)
    start_rss = process.memory_info().private / 1024 / 1024

    # Warmup
    print("Warming up...")
    for _ in range(20):  # 20 warmup runs
        sess.run(None, {input_name: input_data})

    # Memory sample point 2: after warmup
    warmup_rss = process.memory_info().private / 1024 / 1024

    # Benchmark - pure inference, input reused
    print("Running pure inference benchmark...")
    runs = 2000  # 2000 inferences
    times = []
    peak_rss = start_rss

    for i in range(runs):
        # Create a new copy of input tensor each time to avoid CPU cache effects
        # Note: copy() is outside the timing window, only pure inference latency is measured
        input_tensor = input_data.copy()
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        times.append(dt)

        # Sample memory, record peak
        current_rss = process.memory_info().private / 1024 / 1024
        if current_rss > peak_rss:
            peak_rss = current_rss

    # Memory sample point 3: stable value after benchmark
    stable_rss = process.memory_info().private / 1024 / 1024

    # Calculate results
    avg_latency = sum(times) / len(times)
    std_latency = np.std(times)
    min_latency = min(times)
    max_latency = max(times)
    p50_latency = np.percentile(times, 50)
    p90_latency = np.percentile(times, 90)
    p95_latency = np.percentile(times, 95)
    p99_latency = np.percentile(times, 99)  # Extra p99 calculation

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
    print("===== Python Pure Inference Benchmark =====")
    print("Test config:")
    print("- Threads: 12")
    print("- Input data: loaded once, reused in loop")
    print("- Inferences: 2000")
    print("- Warmup: 20 runs")
    print()

    # Test YOLO11x model
    print("\n===== Testing YOLO11x Model =====")
    yolo11x_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
    if not os.path.exists(yolo11x_path):
        print(f"Error: YOLO11x model file not found: {yolo11x_path}")
        sys.exit(1)
    
    yolo11x_result = run_benchmark("YOLO11x", yolo11x_path)
    
    print(f"\nYOLO11x Test Results:")
    print(f"Avg Latency: {yolo11x_result.avg_latency:.3f} ms")
    print(f"Std Dev: {yolo11x_result.std_latency:.3f} ms")
    print(f"P50 Latency: {yolo11x_result.p50_latency:.3f} ms")
    print(f"P90 Latency: {yolo11x_result.p90_latency:.3f} ms")
    print(f"P95 Latency: {yolo11x_result.p95_latency:.3f} ms")
    print(f"Min Latency: {yolo11x_result.min_latency:.3f} ms")
    print(f"Max Latency: {yolo11x_result.max_latency:.3f} ms")
    print(f"Start RSS: {yolo11x_result.start_rss:.2f} MB")
    print(f"Peak RSS: {yolo11x_result.peak_rss:.2f} MB")
    print(f"Stable RSS: {yolo11x_result.stable_rss:.2f} MB")
    print(f"RSS Drift: {yolo11x_result.stable_rss - yolo11x_result.start_rss:.2f} MB")

    # Test YOLO11n model
    print("\n===== Testing YOLO11n Model =====")
    yolo11n_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))
    if not os.path.exists(yolo11n_path):
        print(f"Error: YOLO11n model file not found: {yolo11n_path}")
        sys.exit(1)
    
    yolo11n_result = run_benchmark("YOLO11n", yolo11n_path)
    
    print(f"\nYOLO11n Test Results:")
    print(f"Avg Latency: {yolo11n_result.avg_latency:.3f} ms")
    print(f"Std Dev: {yolo11n_result.std_latency:.3f} ms")
    print(f"P50 Latency: {yolo11n_result.p50_latency:.3f} ms")
    print(f"P90 Latency: {yolo11n_result.p90_latency:.3f} ms")
    print(f"P95 Latency: {yolo11n_result.p95_latency:.3f} ms")
    print(f"Min Latency: {yolo11n_result.min_latency:.3f} ms")
    print(f"Max Latency: {yolo11n_result.max_latency:.3f} ms")
    print(f"Start RSS: {yolo11n_result.start_rss:.2f} MB")
    print(f"Peak RSS: {yolo11n_result.peak_rss:.2f} MB")
    print(f"Stable RSS: {yolo11n_result.stable_rss:.2f} MB")
    print(f"RSS Drift: {yolo11n_result.stable_rss - yolo11n_result.start_rss:.2f} MB")

    # Get system info
    import platform
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu_model = platform.processor()
    system_info = platform.platform()
    
    # Save results
    result_path = os.path.join(base_path, "results", "python_pure_inference_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Pure Inference Benchmark Results =====\n")
        f.write(f"Test time: {timestamp}\n")
        f.write(f"System info: {system_info}\n")
        f.write(f"CPU model: {cpu_model}\n")
        f.write("Test config:\n")
        f.write("- Threads: 12\n")
        f.write("- Input data: loaded once, copy created for each inference\n")
        f.write("- Inferences: 2000\n")
        f.write("- Warmup: 20 runs\n")
        f.write("\n")
        
        f.write("===== YOLO11x Test Results =====\n")
        f.write(f"Model: YOLO11x\n")
        f.write(f"Avg Latency: {yolo11x_result.avg_latency:.5f} ms\n")
        f.write(f"Std Dev: {yolo11x_result.std_latency:.5f} ms\n")
        f.write(f"P50 Latency: {yolo11x_result.p50_latency:.5f} ms\n")
        f.write(f"P90 Latency: {yolo11x_result.p90_latency:.5f} ms\n")
        f.write(f"P95 Latency: {yolo11x_result.p95_latency:.5f} ms\n")
        f.write(f"Min Latency: {yolo11x_result.min_latency:.5f} ms\n")
        f.write(f"Max Latency: {yolo11x_result.max_latency:.5f} ms\n")
        f.write(f"Start RSS: {yolo11x_result.start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {yolo11x_result.peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {yolo11x_result.stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {yolo11x_result.stable_rss - yolo11x_result.start_rss:.5f} MB\n")
        f.write("\n")
        
        f.write("===== YOLO11n Test Results =====\n")
        f.write(f"Model: YOLO11n\n")
        f.write(f"Avg Latency: {yolo11n_result.avg_latency:.5f} ms\n")
        f.write(f"Std Dev: {yolo11n_result.std_latency:.5f} ms\n")
        f.write(f"P50 Latency: {yolo11n_result.p50_latency:.5f} ms\n")
        f.write(f"P90 Latency: {yolo11n_result.p90_latency:.5f} ms\n")
        f.write(f"P95 Latency: {yolo11n_result.p95_latency:.5f} ms\n")
        f.write(f"Min Latency: {yolo11n_result.min_latency:.5f} ms\n")
        f.write(f"Max Latency: {yolo11n_result.max_latency:.5f} ms\n")
        f.write(f"Start RSS: {yolo11n_result.start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {yolo11n_result.peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {yolo11n_result.stable_rss:.5f} MB\n")
        f.write(f"RSS Drift: {yolo11n_result.stable_rss - yolo11n_result.start_rss:.5f} MB\n")

    print(f"\nResults saved to: {result_path}")
    print("Test complete!")

if __name__ == "__main__":
    main()
