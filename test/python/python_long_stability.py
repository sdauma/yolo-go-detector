# -*- coding: utf-8 -*-
# python_long_stability.py
# Python 10-minute stability test - baseline execution path
#
# Important (P0 principle):
# This test uses Python baseline Session API (InferenceSession).
# Python ONNX Runtime's run() method copies data on each call
# (no Go-style I/O Binding). Per P0 principle, this test is for
# observing phenomena only, not for language-level performance conclusions.
#
# Technical:
# - Uses fixed thread config: intra_op_num_threads=12, inter_op_num_threads=1
# - All SessionOptions params explicitly set (P2 principle)
#
# Test purpose:
# - Measure baseline stability over 10 minutes of continuous inference
#   under fixed thread configuration
# - Record RSS memory drift and latency fluctuation indicators
# - Not for language-level performance conclusions

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
import csv
from datetime import datetime

# Fixed random seed for reproducibility
np.random.seed(12345)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Current dir: {current_dir}")

# Build model path
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
print(f"Model path: {model_path}")

# Build project root path
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
print(f"Project root: {base_path}")

# Check model file
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

print("===== Python Long-Term Stability Test =====")
print("Test duration: 10 minutes")
print("Sample interval: 1 second")

# Create Session
print("Creating InferenceSession...")
try:
    sess_options = ort.SessionOptions()

    # Explicitly set all SessionOptions params (P2 principle: no reliance on defaults)
    # Thread config - 12 threads, consistent with other tests
    sess_options.intra_op_num_threads = 12
    sess_options.inter_op_num_threads = 1

    # Log config (disable all logs to avoid log IO interference)
    sess_options.log_severity_level = 3  # 3 = ORT_LOGGING_LEVEL_ERROR

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
print(f"Input name: {input_name}")
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

# Get process object
process = psutil.Process(os.getpid())

# Warmup
print("Warming up...")
for _ in range(10):
    sess.run(None, {input_name: input_data})
print("Warmup complete!")

# Start long-term stability test
print("\n===== Starting Long-Term Stability Test =====")
print("Test duration: 10 minutes (600 seconds)")
print("Sample interval: 1 second")
print("Inference mode: continuous")

# Test params
test_duration = 10 * 60  # 10 minutes in seconds
sample_interval = 1  # 1 second sample interval
start_time = time.time()
end_time = start_time + test_duration

# RSS sample data
rss_samples = []
inference_times = []
peak_rss = 0
min_rss = float('inf')

# Initial RSS sample
initial_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
peak_rss = initial_rss
min_rss = initial_rss
rss_samples.append({
    'timestamp': datetime.now(),
    'elapsed': 0,
    'rss': initial_rss
})
print(f"Initial RSS: {initial_rss:.5f} MB")

# Inference counter
inference_count = 0

# Main test loop
while time.time() < end_time:
    # Execute inference
    t0 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000  # Convert to ms
    inference_times.append(dt)
    inference_count += 1

    # Sample RSS (once per second)
    current_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
    if current_rss > peak_rss:
        peak_rss = current_rss
    if current_rss < min_rss:
        min_rss = current_rss

    elapsed = time.time() - start_time
    rss_samples.append({
        'timestamp': datetime.now(),
        'elapsed': elapsed,
        'rss': current_rss
    })

    # Print progress every minute
    if int(elapsed) % 60 == 0 and int(elapsed) > 0:
        remaining = end_time - time.time()
        print(f"Progress: {inference_count} inferences, elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s, current RSS: {current_rss:.5f} MB")

    # Wait 1 second (note: actual interval = inference time + sleep(1), negligible for long-term tests)
    time.sleep(sample_interval)

# Final RSS sample
final_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
rss_samples.append({
    'timestamp': datetime.now(),
    'elapsed': time.time() - start_time,
    'rss': final_rss
})

# Compute statistics
total_duration = time.time() - start_time
avg_inference_time = np.mean(inference_times)
min_inference_time = np.min(inference_times)
max_inference_time = np.max(inference_times)
p50_inference_time = np.percentile(inference_times, 50)
p90_inference_time = np.percentile(inference_times, 90)
p99_inference_time = np.percentile(inference_times, 99)

# Compute RSS statistics
rss_values = [sample['rss'] for sample in rss_samples]
avg_rss = np.mean(rss_values)
rss_drift = final_rss - initial_rss
rss_range = peak_rss - min_rss
rss_range_percent = (rss_range / avg_rss) * 100 if avg_rss > 0 else 0

# Output test results
print(f"\n===== Long-Term Stability Test Results =====")
print(f"Test duration: {total_duration:.0f}s")
print(f"Inference count: {inference_count}")
print(f"Inference rate: {inference_count / total_duration:.2f} inferences/s")

print(f"\n===== Inference Performance Statistics =====")
print(f"Avg inference time: {avg_inference_time:.5f} ms")
print(f"P50 inference time: {p50_inference_time:.5f} ms")
print(f"P90 inference time: {p90_inference_time:.5f} ms")
print(f"P99 inference time: {p99_inference_time:.5f} ms")
print(f"Min inference time: {min_inference_time:.5f} ms")
print(f"Max inference time: {max_inference_time:.5f} ms")

print(f"\n===== Memory Usage Statistics =====")
print(f"Initial RSS: {initial_rss:.5f} MB")
print(f"Final RSS: {final_rss:.5f} MB")
print(f"Avg RSS: {avg_rss:.5f} MB")
print(f"Peak PM: {peak_rss:.5f} MB")
print(f"Min RSS: {min_rss:.5f} MB")
print(f"PM Drift: {rss_drift:.5f} MB")
print(f"RSS Fluctuation Range: {rss_range:.5f} MB ({rss_range_percent:.2f}%)")

# Save detailed results
result_path = os.path.join(current_dir, '..', '..', 'results', 'python_long_stability_result.txt')
print(f"\nSaving results to: {result_path}")
try:
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Long-Term Stability Test Results =====\n")
        f.write(f"Test duration: {total_duration:.0f}s\n")
        f.write(f"Inference count: {inference_count}\n")
        f.write(f"Inference rate: {inference_count / total_duration:.2f} inferences/s\n")
        f.write(f"\n===== Inference Performance Statistics =====\n")
        f.write(f"Avg inference time: {avg_inference_time:.5f} ms\n")
        f.write(f"P50 inference time: {p50_inference_time:.5f} ms\n")
        f.write(f"P90 inference time: {p90_inference_time:.5f} ms\n")
        f.write(f"P99 inference time: {p99_inference_time:.5f} ms\n")
        f.write(f"Min inference time: {min_inference_time:.5f} ms\n")
        f.write(f"Max inference time: {max_inference_time:.5f} ms\n")
        f.write(f"\n===== Memory Usage Statistics =====\n")
        f.write(f"Initial RSS: {initial_rss:.5f} MB\n")
        f.write(f"Final RSS: {final_rss:.5f} MB\n")
        f.write(f"Avg RSS: {avg_rss:.5f} MB\n")
        f.write(f"Peak PM: {peak_rss:.5f} MB\n")
        f.write(f"Min RSS: {min_rss:.5f} MB\n")
        f.write(f"PM Drift: {rss_drift:.5f} MB\n")
        f.write(f"RSS Fluctuation Range: {rss_range:.5f} MB ({rss_range_percent:.2f}%)\n")
    print("Results saved successfully!")
except Exception as e:
    print(f"Error saving results: {e}")

# Save RSS curve data
rss_data_path = os.path.join(current_dir, '..', '..', 'results', 'python_rss_curve.csv')
print(f"Saving RSS curve data to: {rss_data_path}")
try:
    with open(rss_data_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Elapsed_Seconds', 'RSS_MB'])
        for sample in rss_samples:
            writer.writerow([
                sample['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                f"{sample['elapsed']:.3f}",
                f"{sample['rss']:.5f}"
            ])
    print(f"RSS curve data saved: {len(rss_samples)} sample points")
except Exception as e:
    print(f"Error saving RSS curve data: {e}")

print("\nTest complete!")
