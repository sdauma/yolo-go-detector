# -*- coding: utf-8 -*-
# python_cold_start_benchmark.py
# Python cold start time comparison analysis - Baseline execution path
# 
# Important statement (P0 principle):
# This test uses Python baseline Session API (InferenceSession).
# Python ONNX Runtime's run() method copies data on each call (no Go-style I/O Binding).
# Per P0 principle, this test is for phenomenon observation only, NOT for language-level performance conclusions.
# 
# Technical notes:
# - Uses fixed thread config: intra_op_num_threads=12, inter_op_num_threads=1
# - All SessionOptions params explicitly set (P2 principle)
# Test purpose:
# - Measure cold start time (first inference after Session creation) and stable state time
# - Provide reference data for comparison with Go cold start
# - Not for language-level performance conclusions

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil

# Fixed random seed for reproducibility
np.random.seed(12345)

# Get current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Current directory: {current_dir}")

# Build model path
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
print(f"Model path: {model_path}")

# Build project root path
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
print(f"Project root: {base_path}")

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

print("===== Python Cold Start Time Comparison Analysis =====")
print(f"Model path: {model_path}")

# Run 5 independent tests
test_count = 5
all_cold_start_times = []
all_avg_stable_latencies = []
all_min_stable_latencies = []
all_max_stable_latencies = []
all_p50_stable_latencies = []
all_p90_stable_latencies = []
all_p99_stable_latencies = []
all_start_rss = []
all_cold_start_rss = []
all_stable_rss = []

for test_idx in range(1, test_count + 1):
    print(f"\n=== Independent Test {test_idx}/{test_count} ===")

    # Create Session
    print("Creating InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # Explicitly set all SessionOptions params (P2 principle: forbid relying on defaults)
        # Thread config - 12 threads, consistent with other tests
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        
        # Log config (disable all logs to avoid log I/O interfering with performance)
        sess_options.log_severity_level = 3  # 3 = ORT_LOGGING_LEVEL_ERROR
        
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

    # Use input data completely consistent with Go (loaded from file)
    print("Loading input data...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"Input data loaded successfully: {input_data_path}")
    except Exception as e:
        print(f"Failed to load input data: {e}")
        sys.exit(1)

    # Memory sample point 1: after Session creation (Start RSS)
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
    print(f"Start RSS: {start_rss:.5f} MB")

    # Test cold start time
    print("\n===== Testing Cold Start Time =====")
    t0 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t1 = time.perf_counter()
    cold_start_time = (t1 - t0) * 1000.0
    print(f"Cold Start Time: {cold_start_time:.5f} ms")

    # Memory sample point 2: after cold start (Cold Start RSS)
    cold_start_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
    print(f"Cold Start RSS: {cold_start_rss:.5f} MB")

    # Warmup phase
    print("\n===== Warmup Phase =====")
    warmup_count = 10
    warmup_latencies = []
    for i in range(warmup_count):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        warmup_latencies.append(dt)

    # Stable state test
    print("\n===== Stable State Test =====")
    stable_count = 100
    stable_latencies = []
    peak_rss = cold_start_rss

    for i in range(stable_count):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        stable_latencies.append(dt)

        # Sample memory every 10 inferences, record peak
        if i % 10 == 0:
            current_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
            if current_rss > peak_rss:
                peak_rss = current_rss

    # Memory sample point 3: after stable state (Stable RSS)
    stable_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
    print(f"\nStable RSS: {stable_rss:.5f} MB")
    print(f"Peak RSS: {peak_rss:.5f} MB")

    # Calculate stable state statistics
    avg_stable_latency = sum(stable_latencies) / len(stable_latencies)
    min_stable_latency = min(stable_latencies)
    max_stable_latency = max(stable_latencies)
    p50_stable_latency = np.percentile(stable_latencies, 50)
    p90_stable_latency = np.percentile(stable_latencies, 90)
    p99_stable_latency = np.percentile(stable_latencies, 99)

    # Save this test's results
    all_cold_start_times.append(cold_start_time)
    all_avg_stable_latencies.append(avg_stable_latency)
    all_min_stable_latencies.append(min_stable_latency)
    all_max_stable_latencies.append(max_stable_latency)
    all_p50_stable_latencies.append(p50_stable_latency)
    all_p90_stable_latencies.append(p90_stable_latency)
    all_p99_stable_latencies.append(p99_stable_latency)
    all_start_rss.append(start_rss)
    all_cold_start_rss.append(cold_start_rss)
    all_stable_rss.append(stable_rss)

    print(f"Test {test_idx} complete: Cold Start={cold_start_time:.5f} ms, Avg Stable={avg_stable_latency:.5f} ms")

# Calculate average of 5 tests
cold_start_time = np.mean(all_cold_start_times)
avg_stable_latency = np.mean(all_avg_stable_latencies)
min_stable_latency = np.mean(all_min_stable_latencies)
max_stable_latency = np.mean(all_max_stable_latencies)
p50_stable_latency = np.mean(all_p50_stable_latencies)
p90_stable_latency = np.mean(all_p90_stable_latencies)
p99_stable_latency = np.mean(all_p99_stable_latencies)
start_rss = np.mean(all_start_rss)
cold_start_rss = np.mean(all_cold_start_rss)
stable_rss = np.mean(all_stable_rss)

# Calculate std dev
std_dev_stable = np.std(all_avg_stable_latencies)
# Calculate coefficient of variation
coeff_var_stable = (std_dev_stable / avg_stable_latency) * 100
# Calculate FPS
fps = 1000.0 / avg_stable_latency

# Output results
print("\n===== Cold Start vs Stable State Comparison =====")
print(f"Cold Start Time: {cold_start_time:.5f} ms")
print(f"Avg Stable Latency: {avg_stable_latency:.5f} ms")
print(f"Cold Start / Stable Ratio: {cold_start_time/avg_stable_latency:.2f}x")
print("\n===== Stable State Detailed Stats =====")
print(f"Avg Latency: {avg_stable_latency:.5f} ms")
print(f"Std Dev: {std_dev_stable:.5f} ms")
print(f"Coeff of Variation: {coeff_var_stable:.2f}%")
print(f"FPS: {fps:.2f}")
print(f"Min Latency: {min_stable_latency:.5f} ms")
print(f"Max Latency: {max_stable_latency:.5f} ms")
print(f"P50 Latency: {p50_stable_latency:.5f} ms")
print(f"P90 Latency: {p90_stable_latency:.5f} ms")
print(f"P99 Latency: {p99_stable_latency:.5f} ms")
print("\n===== Memory Usage =====")
print(f"Start RSS: {start_rss:.5f} MB")
print(f"Cold Start RSS: {cold_start_rss:.5f} MB")
print(f"Stable RSS: {stable_rss:.5f} MB")
print(f"Memory Growth (Start -> Cold Start): {cold_start_rss-start_rss:.5f} MB")
print(f"Memory Growth (Cold Start -> Stable): {stable_rss-cold_start_rss:.5f} MB")

# Save detailed log
log_path = os.path.join(current_dir, '..', '..', 'results', 'python_cold_start_detailed_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    for i in range(len(all_cold_start_times)):
        f.write(f"===== Test Run #{i+1} =====\n")
        f.write(f"Cold Start Time: {all_cold_start_times[i]:.5f} ms\n")
        f.write(f"Avg Stable Latency: {all_avg_stable_latencies[i]:.5f} ms\n")
        f.write(f"Min Latency: {all_min_stable_latencies[i]:.5f} ms\n")
        f.write(f"Max Latency: {all_max_stable_latencies[i]:.5f} ms\n")
        f.write(f"P50 Latency: {all_p50_stable_latencies[i]:.5f} ms\n")
        f.write(f"P90 Latency: {all_p90_stable_latencies[i]:.5f} ms\n")
        f.write(f"P99 Latency: {all_p99_stable_latencies[i]:.5f} ms\n")
        f.write(f"Start RSS: {all_start_rss[i]:.5f} MB\n")
        f.write(f"Cold Start RSS: {all_cold_start_rss[i]:.5f} MB\n")
        f.write(f"Stable RSS: {all_stable_rss[i]:.5f} MB\n")
        f.write("\n")

    f.write("===== 5-Test Average =====\n")
    f.write(f"Cold Start Time: {cold_start_time:.5f} ms\n")
    f.write(f"Avg Stable Latency: {avg_stable_latency:.5f} ms\n")
    f.write(f"Cold Start / Stable Ratio: {cold_start_time/avg_stable_latency:.2f}x\n\n")

    f.write("===== Stable State Detailed Stats =====\n")
    f.write(f"Avg Latency: {avg_stable_latency:.5f} ms\n")
    f.write(f"Std Dev: {std_dev_stable:.5f} ms\n")
    f.write(f"Coeff of Variation: {coeff_var_stable:.2f}%\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"Min Latency: {min_stable_latency:.5f} ms\n")
    f.write(f"Max Latency: {max_stable_latency:.5f} ms\n")
    f.write(f"P50 Latency: {p50_stable_latency:.5f} ms\n")
    f.write(f"P90 Latency: {p90_stable_latency:.5f} ms\n")
    f.write(f"P99 Latency: {p99_stable_latency:.5f} ms\n")

    f.write("\n===== Memory Usage =====\n")
    f.write(f"Start RSS: {start_rss:.5f} MB\n")
    f.write(f"Cold Start RSS: {cold_start_rss:.5f} MB\n")
    f.write(f"Stable RSS: {stable_rss:.5f} MB\n")
    f.write(f"Memory Growth (Start -> Cold Start): {cold_start_rss-start_rss:.5f} MB\n")
    f.write(f"Memory Growth (Cold Start -> Stable): {stable_rss-cold_start_rss:.5f} MB\n")

print(f"\nDetailed log saved to: {log_path}")

# Save results
result_path = os.path.join(current_dir, '..', '..', 'results', 'python_cold_start_result.txt')
print(f"\nSaving results to: {result_path}")

# Build result strings
result_lines = [
    "===== Python Cold Start Time Comparison Analysis (5-run average) =====",
    f"Cold Start Time: {cold_start_time:.5f} ms",
    f"Avg Stable Latency: {avg_stable_latency:.5f} ms",
    f"Cold Start / Stable Ratio: {cold_start_time/avg_stable_latency:.2f}x",
    "",
    "===== Stable State Detailed Stats =====",
    f"Avg Latency: {avg_stable_latency:.5f} ms",
    f"Std Dev: {std_dev_stable:.5f} ms",
    f"Coeff of Variation: {coeff_var_stable:.2f}%",
    f"FPS: {fps:.2f}",
    f"Min Latency: {min_stable_latency:.5f} ms",
    f"Max Latency: {max_stable_latency:.5f} ms",
    f"P50 Latency: {p50_stable_latency:.5f} ms",
    f"P90 Latency: {p90_stable_latency:.5f} ms",
    f"P99 Latency: {p99_stable_latency:.5f} ms",
    "",
    "===== Memory Usage =====",
    f"Start RSS: {start_rss:.5f} MB",
    f"Cold Start RSS: {cold_start_rss:.5f} MB",
    f"Stable RSS: {stable_rss:.5f} MB",
    f"Memory Growth (Start -> Cold Start): {cold_start_rss-start_rss:.5f} MB",
    f"Memory Growth (Cold Start -> Stable): {stable_rss-cold_start_rss:.5f} MB"
]

# Write with UTF-8 encoding
with open(result_path, 'w', encoding='utf-8') as f:
    for line in result_lines:
        f.write(line + '\n')

print(f"\nResults saved to: {result_path}")
print("\n===== Cold Start Time Comparison Analysis Complete =====")
