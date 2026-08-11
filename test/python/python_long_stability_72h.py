
# -*- coding: utf-8 -*-
# python_long_stability_72h.py
# Python 72-Hour Long-Term Stability Test
# Important (P0 principle):
# This test uses Python baseline Session interface (InferenceSession).
# Python ONNX Runtime's run() method performs data copy on each call
# (no Go-style I/O Binding). Per P0 principle, this test is only for
# observing phenomena, not for language-level performance conclusions.
#
# Technical:
# - Uses fixed thread config: intra_op_num_threads=12, inter_op_num_threads=1
# - All SessionOptions params explicitly set (P2 principle)
# - 100 inferences per hour, continuous for 72 hours
#
# Purpose:
# - Verify single-Session baseline stability of Python ONNX Runtime
#   over 72 hours of continuous operation
# - Aligned with Go 72h test (go_long_stability_72h.go), providing
#   dual-language comparison data

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
import json
import csv
from datetime import datetime

# Fixed random seed for reproducibility
np.random.seed(12345)

# ========== Configuration (aligned with Go 72h) ==========
# Supports command line argument: python python_long_stability_72h.py [hours]
# Default 72 hours
TEST_DURATION_HOURS = 72
if len(sys.argv) > 1:
    try:
        TEST_DURATION_HOURS = float(sys.argv[1])
        if TEST_DURATION_HOURS <= 0:
            raise ValueError("Duration must be positive")
    except ValueError:
        print(f"Error: Invalid hours argument '{sys.argv[1]}', using default 72")
        TEST_DURATION_HOURS = 72
INFERENCES_PER_HOUR = 100
INFERENCE_INTERVAL_SEC = 0.5  # 500ms between inferences
WARMUP_INFERENCES = 20
RSS_SAMPLE_INTERVAL = 10  # Sample RSS every N inferences

# ========== Paths ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
model_path = os.path.abspath(os.path.join(base_path, 'third_party', 'yolo11x.onnx'))
input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
results_dir = os.path.join(base_path, "results")
os.makedirs(results_dir, exist_ok=True)

json_path = os.path.join(results_dir, f"python_stability_{TEST_DURATION_HOURS:.0f}h_result.json")
csv_path = os.path.join(results_dir, f"python_stability_{TEST_DURATION_HOURS:.0f}h_detailed.csv")
txt_path = os.path.join(results_dir, f"python_stability_{TEST_DURATION_HOURS:.0f}h_result.txt")

# ========== Check model ==========
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

test_duration_seconds = TEST_DURATION_HOURS * 3600
total_inferences_expected = TEST_DURATION_HOURS * INFERENCES_PER_HOUR

print("=" * 70)
print("  Python 72-Hour Long-Term Stability Test")
print("=" * 70)
print(f"  Model:          yolo11x.onnx")
print(f"  Duration:       {TEST_DURATION_HOURS} hours (time-controlled, aligned with Go)")
print(f"  Inferences/hr:  {INFERENCES_PER_HOUR}")
print(f"  Expected total: ~{total_inferences_expected} inferences")
print(f"  Interval:       {INFERENCE_INTERVAL_SEC}s")
print(f"  Warmup:         {WARMUP_INFERENCES} inferences")
print(f"  Threads:        intra=12, inter=1")
print(f"  Start time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ========== Create Session ==========
print("\n[1/4] Creating InferenceSession...")
try:
    sess_options = ort.SessionOptions()

    # Explicitly set all SessionOptions params (P2 principle: no reliance on defaults)
    sess_options.intra_op_num_threads = 12
    sess_options.inter_op_num_threads = 1

    # Log config (disable all logs to avoid log IO interference)
    sess_options.log_severity_level = 3  # 3 = ORT_LOGGING_LEVEL_ERROR

    # Execution mode config
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Memory pool config
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # All unmentioned Session params use ONNX Runtime 1.23.2 official defaults
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    print("  InferenceSession created successfully!")
except Exception as e:
    print(f"  ERROR: Failed to create InferenceSession: {e}")
    sys.exit(1)

# ========== Load input data ==========
print("\n[2/4] Loading input data...")
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"  Input name: {input_name}")
print(f"  Input shape: {input_shape}")

try:
    input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
    print(f"  Loaded from {input_data_path}")
except Exception as e:
    print(f"  Failed to load input data: {e}")
    sys.exit(1)

# ========== Warmup ==========
print(f"\n[3/4] Running {WARMUP_INFERENCES} warmup inferences...")
process = psutil.Process(os.getpid())
for i in range(WARMUP_INFERENCES):
    sess.run(None, {input_name: input_data})
    if (i + 1) % 5 == 0:
        print(f"  Warmup: {i+1}/{WARMUP_INFERENCES}")
print("  Warmup complete.")

# ========== Main Test Loop ==========
print(f"\n[4/4] Starting 72-hour test (time-controlled, ~{total_inferences_expected} inferences)...")
print("-" * 70)

start_time = time.time()

# Initial RSS
start_rss = process.memory_info().private / 1024 / 1024
peak_rss = start_rss
min_rss = start_rss

# Tracking variables
latencies = []
rss_samples = []  # (hour, rss_mb)
hourly_latencies = []  # (hour, avg_latency_ms)

# For hourly reporting
next_report_hour = 1
hourly_latency_buffer = []

total_inferences = 0
errors = 0
last_status_time = start_time


def format_duration(seconds):
    """Format seconds into H:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


try:
    while (time.time() - start_time) < test_duration_seconds:
        # Run inference
        t0 = time.perf_counter()
        try:
            sess.run(None, {input_name: input_data})
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            hourly_latency_buffer.append(latency_ms)
        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            if errors <= 5:
                print(f"  ERROR at inference #{total_inferences+1}: {e}")

        total_inferences += 1

        # Sample RSS every N inferences
        if total_inferences % RSS_SAMPLE_INTERVAL == 0:
            current_rss = process.memory_info().private / 1024 / 1024
            if current_rss > peak_rss:
                peak_rss = current_rss
            if current_rss < min_rss:
                min_rss = current_rss
            elapsed_hours = (time.time() - start_time) / 3600
            rss_samples.append((elapsed_hours, current_rss))

        # Hourly status report
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= next_report_hour and len(hourly_latency_buffer) > 0:
            avg_lat = np.mean(hourly_latency_buffer)
            p50_lat = np.percentile(hourly_latency_buffer, 50)
            p99_lat = np.percentile(hourly_latency_buffer, 99)
            hourly_latencies.append((elapsed_hours, avg_lat))
            current_rss = process.memory_info().private / 1024 / 1024

            progress_pct = min(elapsed_hours / TEST_DURATION_HOURS * 100, 100)
            elapsed_str = format_duration(elapsed_hours * 3600)
            print(f"  [{elapsed_str}] "
                  f"Inferences: {total_inferences} | "
                  f"Progress: {progress_pct:.1f}% | "
                  f"RSS: {current_rss:.1f} MB | "
                  f"Latency: avg={avg_lat:.1f}ms p50={p50_lat:.1f}ms p99={p99_lat:.1f}ms")

            hourly_latency_buffer = []
            next_report_hour += 1

        # Status dot every 10 minutes (during long silent periods)
        if time.time() - last_status_time >= 600:  # 10 minutes
            current_rss = process.memory_info().private / 1024 / 1024
            elapsed_hours = (time.time() - start_time) / 3600
            elapsed_str = format_duration(elapsed_hours * 3600)
            progress_pct = min(elapsed_hours / TEST_DURATION_HOURS * 100, 100)
            print(f"  [{elapsed_str}] Status: {total_inferences} inferences, "
                  f"Progress: {progress_pct:.1f}%, RSS: {current_rss:.1f} MB, Errors: {errors}")
            last_status_time = time.time()

        # Sleep between inferences
        time.sleep(INFERENCE_INTERVAL_SEC)

except KeyboardInterrupt:
    print("\n\n  Test interrupted by user.")

# ========== Final Measurements ==========
end_time = time.time()
end_rss = process.memory_info().private / 1024 / 1024
actual_duration_hours = (end_time - start_time) / 3600
rss_drift = end_rss - start_rss
rss_drift_rate = rss_drift / actual_duration_hours if actual_duration_hours > 0 else 0

# Compute statistics
if latencies:
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
else:
    avg_latency = p50_latency = p95_latency = p99_latency = std_latency = min_latency = max_latency = 0

# ========== Results ==========
print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
print(f"  Duration:       {actual_duration_hours:.2f} hours ({format_duration(end_time-start_time)})")
print(f"  Inferences:     {total_inferences}")
print(f"  Errors:         {errors}")
print(f"  Start PM:      {start_rss:.2f} MB")
print(f"  End RSS:        {end_rss:.2f} MB")
print(f"  Peak PM:       {peak_rss:.2f} MB")
print(f"  Min RSS:        {min_rss:.2f} MB")
print(f"  PM Drift:      {rss_drift:.2f} MB")
print(f"  Drift Rate:     {rss_drift_rate:.4f} MB/hour")
print(f"  Avg Latency:    {avg_latency:.2f} ms")
print(f"  P50 Latency:    {p50_latency:.2f} ms")
print(f"  P95 Latency:    {p95_latency:.2f} ms")
print(f"  P99 Latency:    {p99_latency:.2f} ms")
print(f"  Std Latency:    {std_latency:.2f} ms")
print(f"  Min Latency:    {min_latency:.2f} ms")
print(f"  Max Latency:    {max_latency:.2f} ms")
print("=" * 70)

# ========== Save JSON ==========
json_result = {
    "test_name": "python_long_stability_72h",
    "model": "yolo11x.onnx",
    "config": {
        "test_duration_hours": TEST_DURATION_HOURS,
        "inferences_per_hour": INFERENCES_PER_HOUR,
        "inference_interval_sec": INFERENCE_INTERVAL_SEC,
        "warmup_inferences": WARMUP_INFERENCES,
        "intra_op_threads": 12,
        "inter_op_threads": 1,
    },
    "results": {
        "actual_duration_hours": round(actual_duration_hours, 4),
        "total_inferences": total_inferences,
        "errors": errors,
        "start_pm_mb": round(start_rss, 2),
        "end_pm_mb": round(end_rss, 2),
        "peak_pm_mb": round(peak_rss, 2),
        "min_pm_mb": round(min_rss, 2),
        "pm_drift_mb": round(rss_drift, 2),
        "rss_drift_rate_mb_per_hour": round(rss_drift_rate, 4),
        "latency": {
            "avg_ms": round(avg_latency, 2),
            "p50_ms": round(p50_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "p99_ms": round(p99_latency, 2),
            "std_ms": round(std_latency, 2),
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2),
        },
    },
    "pm_samples": [{"hour": round(h, 4), "pm_mb": round(r, 2)} for h, r in rss_samples],
    "hourly_latencies": [{"hour": round(h, 4), "avg_latency_ms": round(l, 2)} for h, l in hourly_latencies],
    "timestamp": datetime.now().isoformat(),
}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_result, f, indent=2, ensure_ascii=False)
print(f"\n  JSON saved: {json_path}")

# ========== Save CSV (RSS samples) ==========
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["hour", "pm_mb"])
    for h, r in rss_samples:
        writer.writerow([round(h, 4), round(r, 2)])
print(f"  CSV saved:  {csv_path}")

# ========== Save TXT Summary ==========
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("Python 72-Hour Long-Term Stability Test Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model:            yolo11x.onnx\n")
    f.write(f"Duration:         {actual_duration_hours:.2f} hours\n")
    f.write(f"Total Inferences: {total_inferences}\n")
    f.write(f"Errors:           {errors}\n")
    f.write(f"Start PM:        {start_rss:.2f} MB\n")
    f.write(f"End RSS:          {end_rss:.2f} MB\n")
    f.write(f"Peak PM:         {peak_rss:.2f} MB\n")
    f.write(f"Min RSS:          {min_rss:.2f} MB\n")
    f.write(f"PM Drift:        {rss_drift:.2f} MB\n")
    f.write(f"Drift Rate:       {rss_drift_rate:.4f} MB/hour\n")
    f.write(f"Avg Latency:      {avg_latency:.2f} ms\n")
    f.write(f"P50 Latency:      {p50_latency:.2f} ms\n")
    f.write(f"P95 Latency:      {p95_latency:.2f} ms\n")
    f.write(f"P99 Latency:      {p99_latency:.2f} ms\n")
    f.write(f"Std Latency:      {std_latency:.2f} ms\n")
    f.write(f"Min Latency:      {min_latency:.2f} ms\n")
    f.write(f"Max Latency:      {max_latency:.2f} ms\n")
print(f"  TXT saved:  {txt_path}")

print(f"\n  All results saved to: {results_dir}")
print("  Test finished successfully.")
