# -*- coding: utf-8 -*-
# python_thread_config_benchmark.py
# Python thread config performance benchmark
#
# Technical notes:
# - Uses Python baseline Session API (InferenceSession)
# - Explicitly configures intra_op_num_threads via SessionOptions
# - Tests thread configs: 1, 2, 4, 8
#
# Test purpose:
# - Observe performance trends under different thread configs
# - Verify ONNX Runtime thread scalability
# - Align with Go thread config tests
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

print("===== Python Thread Config Performance Benchmark =====")
print(f"Model path: {model_path}")

# Thread configs to test
thread_configs = [1, 2, 4, 8]

# Store comprehensive results for all thread configs
all_thread_results = []

# Test each thread config
for num_threads in thread_configs:
    print(f"\n===== Testing Thread Config: intra_op_num_threads={num_threads} =====")
    
    # Run 5 independent tests
    test_count = 5
    all_avg_latencies = []
    all_min_latencies = []
    all_max_latencies = []
    all_p50_latencies = []
    all_p90_latencies = []
    all_p99_latencies = []
    all_start_rss = []
    all_peak_rss = []
    all_stable_rss = []
    
    for test_idx in range(1, test_count + 1):
        print(f"\n=== Independent Test {test_idx}/{test_count} ===")
        
        # Create Session
        print("Creating InferenceSession...")
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = 1
            
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            print("InferenceSession created successfully!")
        except Exception as e:
            print(f"Error: Failed to create InferenceSession: {e}")
            continue
        
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
        
        # Memory sample point 1: after Session creation, before warmup (Start RSS)
        process = psutil.Process(os.getpid())
        start_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
        print(f"Start RSS: {start_rss:.5f} MB")
        
        # Warmup
        print("Warming up...")
        for i in range(10):
            t0 = time.perf_counter()
            sess.run(None, {input_name: input_data})
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000.0
        
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
            
            # Sample memory, record peak
            current_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
            if current_rss > peak_rss:
                peak_rss = current_rss
        
        # Memory sample point 3: stable value after benchmark
        stable_rss = process.memory_info().private / 1024 / 1024  # Convert to MB
        print(f"Stable RSS: {stable_rss:.5f} MB")
        print(f"Peak RSS: {peak_rss:.5f} MB")
        
        # Calculate results
        avg_latency = sum(times) / len(times)
        min_latency = min(times)
        max_latency = max(times)
        p50_latency = np.percentile(times, 50)
        p90_latency = np.percentile(times, 90)
        p99_latency = np.percentile(times, 99)
        
        # Save this test's results
        all_avg_latencies.append(avg_latency)
        all_min_latencies.append(min_latency)
        all_max_latencies.append(max_latency)
        all_p50_latencies.append(p50_latency)
        all_p90_latencies.append(p90_latency)
        all_p99_latencies.append(p99_latency)
        all_start_rss.append(start_rss)
        all_peak_rss.append(peak_rss)
        all_stable_rss.append(stable_rss)
        
        print(f"Test {test_idx} complete: Avg Latency={avg_latency:.5f} ms")
    
    # Calculate average of 5 tests
    avg_latency = np.mean(all_avg_latencies)
    min_latency = np.mean(all_min_latencies)
    max_latency = np.mean(all_max_latencies)
    p50_latency = np.mean(all_p50_latencies)
    p90_latency = np.mean(all_p90_latencies)
    p99_latency = np.mean(all_p99_latencies)
    start_rss = np.mean(all_start_rss)
    peak_rss = np.mean(all_peak_rss)
    stable_rss = np.mean(all_stable_rss)
    
    # Calculate std dev
    std_dev = np.std(all_avg_latencies)
    # Calculate coefficient of variation
    coeff_var = (std_dev / avg_latency) * 100
    # Calculate FPS
    fps = 1000.0 / avg_latency
    
    # Store result in comprehensive results list
    all_thread_results.append({
        'num_threads': num_threads,
        'avg_latency': avg_latency,
        'std_dev': std_dev,
        'coeff_var': coeff_var,
        'fps': fps,
        'p50_latency': p50_latency,
        'p90_latency': p90_latency,
        'p99_latency': p99_latency,
        'start_rss': start_rss,
        'stable_rss': stable_rss
    })
    
    print("\n===== Test Results =====")
    print(f"Avg Latency: {avg_latency:.5f} ms")
    print(f"Std Dev: {std_dev:.5f} ms")
    print(f"Coeff of Variation: {coeff_var:.2f}%")
    print(f"FPS: {fps:.2f}")
    print(f"P50 Latency: {p50_latency:.5f} ms")
    print(f"P90 Latency: {p90_latency:.5f} ms")
    print(f"P99 Latency: {p99_latency:.5f} ms")
    print(f"Min Latency: {min_latency:.5f} ms")
    print(f"Max Latency: {max_latency:.5f} ms")
    print(f"\n===== Memory Usage =====")
    print(f"Start RSS: {start_rss:.5f} MB")
    print(f"Peak RSS: {peak_rss:.5f} MB")
    print(f"Stable RSS: {stable_rss:.5f} MB")
    
    # Save detailed log
    log_path = os.path.join(current_dir, '..', '..', 'results', f'python_thread_{num_threads}_detailed_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        for i in range(len(all_avg_latencies)):
            f.write(f"===== Run #{i+1} =====\n")
            f.write(f"Avg Latency: {all_avg_latencies[i]:.5f} ms\n")
            f.write(f"Min Latency: {all_min_latencies[i]:.5f} ms\n")
            f.write(f"Max Latency: {all_max_latencies[i]:.5f} ms\n")
            f.write(f"P50 Latency: {all_p50_latencies[i]:.5f} ms\n")
            f.write(f"P90 Latency: {all_p90_latencies[i]:.5f} ms\n")
            f.write(f"P99 Latency: {all_p99_latencies[i]:.5f} ms\n")
            f.write(f"Start RSS: {all_start_rss[i]:.5f} MB\n")
            f.write(f"Peak RSS: {all_peak_rss[i]:.5f} MB\n")
            f.write(f"Stable RSS: {all_stable_rss[i]:.5f} MB\n")
            f.write("\n")

        f.write("===== 5-Test Average =====\n")
        f.write(f"Avg Latency: {avg_latency:.5f} ms\n")
        f.write(f"Std Dev: {std_dev:.5f} ms\n")
        f.write(f"Coeff of Variation: {coeff_var:.2f}%\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"P50 Latency: {p50_latency:.5f} ms\n")
        f.write(f"P90 Latency: {p90_latency:.5f} ms\n")
        f.write(f"P99 Latency: {p99_latency:.5f} ms\n")
        f.write(f"Min Latency: {min_latency:.5f} ms\n")
        f.write(f"Max Latency: {max_latency:.5f} ms\n")
        f.write("\n===== Memory Usage =====\n")
        f.write(f"Start RSS: {start_rss:.5f} MB\n")
        f.write(f"Peak RSS: {peak_rss:.5f} MB\n")
        f.write(f"Stable RSS: {stable_rss:.5f} MB\n")

    print(f"\nDetailed log saved to: {log_path}")

    # Save results
    result_path = os.path.join(current_dir, '..', '..', 'results', f'python_thread_{num_threads}_result.txt')
    print(f"Saving results to: {result_path}")
    
    # Build result strings
    result_lines = [
        f"===== Python Thread Config Benchmark Results (5-run average) (intra_op_num_threads={num_threads}) =====",
        f"Avg Latency: {avg_latency:.5f} ms",
        f"Std Dev: {std_dev:.5f} ms",
        f"Coeff of Variation: {coeff_var:.2f}%",
        f"FPS: {fps:.2f}",
        f"P50 Latency: {p50_latency:.5f} ms",
        f"P90 Latency: {p90_latency:.5f} ms",
        f"P99 Latency: {p99_latency:.5f} ms",
        f"Min Latency: {min_latency:.5f} ms",
        f"Max Latency: {max_latency:.5f} ms",
        "",
        "===== Memory Usage =====",
        f"Start RSS: {start_rss:.5f} MB",
        f"Peak RSS: {peak_rss:.5f} MB",
        f"Stable RSS: {stable_rss:.5f} MB"
    ]
    
    # Write with UTF-8 encoding
    with open(result_path, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')
    
    print(f"\nResults saved to: {result_path}")

# Save comprehensive results for all thread configs
comprehensive_result_path = os.path.join(current_dir, '..', '..', 'results', 'python_thread_config_comprehensive.txt')
print(f"\nSaving comprehensive results to: {comprehensive_result_path}")

with open(comprehensive_result_path, 'w', encoding='utf-8') as f:
    f.write("===== Comprehensive Thread Config Performance Benchmark Results =====\n\n")
    f.write(f"{'Thread Config':<20} {'Avg Latency(ms)':<15} {'Std Dev(ms)':<12} {'CV(%)':<12} {'FPS':<10} {'P50 Latency(ms)':<15} {'P90 Latency(ms)':<15} {'P99 Latency(ms)':<15} {'Start RSS(MB)':<15} {'Stable RSS(MB)':<15}\n")
    
    for result in all_thread_results:
        f.write(f"{result['num_threads']:<20} {result['avg_latency']:<15.3f} {result['std_dev']:<12.3f} {result['coeff_var']:<12.2f} {result['fps']:<10.2f} {result['p50_latency']:<15.3f} {result['p90_latency']:<15.3f} {result['p99_latency']:<15.3f} {result['start_rss']:<15.2f} {result['stable_rss']:<15.2f}\n")

print("Comprehensive results file written successfully!")

print("\n===== All Thread Config Tests Complete =====")
