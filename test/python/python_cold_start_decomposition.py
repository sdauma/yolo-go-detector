# -*- coding: utf-8 -*-
# python_cold_start_decomposition.py
# Python cold start decomposition test
#
# Technical notes:
# - Uses Python baseline Session API (InferenceSession)
# - Explicitly configures thread params via SessionOptions (intraOp=12, interOp=1)
# - Uses sess.run() standard call path, no I/O Binding
# - Binds to first 4 CPU cores (cpu_affinity=[0,1,2,3])
#
# Test purpose:
# - Decompose cold start time into session creation time, model loading time, and first inference time
# - Run 20 cold start tests, calculate averages
# - Ensure data stability and reproducibility
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

# Build model paths - test both large model and small model
model_path_large = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
model_path_small = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# Build project root path
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Check if model files exist
if not os.path.exists(model_path_large):
    print(f"Error: Large model file not found: {model_path_large}")
    sys.exit(1)
if not os.path.exists(model_path_small):
    print(f"Error: Small model file not found: {model_path_small}")
    sys.exit(1)

@dataclass
class ColdStartResult:
    session_creation_time: float
    model_loading_time: float
    first_inference_time: float
    total_cold_start_time: float
    start_rss: float
    peak_rss: float

def run_cold_start_test(model_path, model_name):
    print(f"\n===== Python Cold Start Test - {model_name} ====")
    
    # Bind CPU cores
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1, 2, 3])
    
    # Record initial memory
    start_rss = process.memory_info().private / 1024 / 1024
    
    # 1. Session creation time
    print("Creating InferenceSession...")
    t0 = time.perf_counter()
    
    try:
        sess_options = ort.SessionOptions()
        
        # Explicitly set all SessionOptions params
        # Thread config - 12 threads, consistent with other tests
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"Error: Failed to create InferenceSession: {e}")
        sys.exit(1)
    
    t1 = time.perf_counter()
    session_creation_time = (t1 - t0) * 1000
    
    # 2. Model loading time (session creation already includes model loading)
    # Note: In ONNX Runtime, model loading is completed during session creation
    model_loading_time = session_creation_time
    
    # Get input info
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Use input data completely consistent with Go (loaded from file, using fixed seed)
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
    except Exception as e:
        print(f"Failed to load input data: {e}")
        sys.exit(1)

    # 3. First inference time
    print("Running first inference...")
    t2 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t3 = time.perf_counter()
    first_inference_time = (t3 - t2) * 1000
    
    # Calculate total cold start time
    total_cold_start_time = session_creation_time + first_inference_time
    
    # Record peak memory
    peak_rss = process.memory_info().private / 1024 / 1024
    
    print(f"Session Creation Time: {session_creation_time:.3f} ms")
    print(f"Model Loading Time: {model_loading_time:.3f} ms")
    print(f"First Inference Time: {first_inference_time:.3f} ms")
    print(f"Total Cold Start Time: {total_cold_start_time:.3f} ms")
    print(f"Start RSS: {start_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    
    return ColdStartResult(
        session_creation_time=session_creation_time,
        model_loading_time=model_loading_time,
        first_inference_time=first_inference_time,
        total_cold_start_time=total_cold_start_time,
        start_rss=start_rss,
        peak_rss=peak_rss
    )

def main():
    print("===== Python Cold Start Decomposition Test (20 runs) =====")

    # Run 20 tests - large model
    num_runs = 20
    results_large = []
    results_small = []

    print("\n===== Testing Large Model (YOLO11x) =====")
    for i in range(num_runs):
        print(f"\n===== Run #{i+1} =====")
        result = run_cold_start_test(model_path_large, "YOLO11x")
        results_large.append(result)

    print("\n===== Testing Small Model (YOLO11n) =====")
    for i in range(num_runs):
        print(f"\n===== Run #{i+1} =====")
        result = run_cold_start_test(model_path_small, "YOLO11n")
        results_small.append(result)

    # Calculate large model averages
    avg_session_creation_large = sum(r.session_creation_time for r in results_large) / num_runs
    avg_model_loading_large = sum(r.model_loading_time for r in results_large) / num_runs
    avg_first_inference_large = sum(r.first_inference_time for r in results_large) / num_runs
    avg_total_cold_start_large = sum(r.total_cold_start_time for r in results_large) / num_runs
    avg_start_rss_large = sum(r.start_rss for r in results_large) / num_runs
    avg_peak_rss_large = sum(r.peak_rss for r in results_large) / num_runs

    # Calculate small model averages
    avg_session_creation_small = sum(r.session_creation_time for r in results_small) / num_runs
    avg_model_loading_small = sum(r.model_loading_time for r in results_small) / num_runs
    avg_first_inference_small = sum(r.first_inference_time for r in results_small) / num_runs
    avg_total_cold_start_small = sum(r.total_cold_start_time for r in results_small) / num_runs
    avg_start_rss_small = sum(r.start_rss for r in results_small) / num_runs
    avg_peak_rss_small = sum(r.peak_rss for r in results_small) / num_runs

    print("\n===== Large Model (YOLO11x) 20-Run Average =====")
    print(f"Session Creation Time: {avg_session_creation_large:.3f} ms")
    print(f"Model Loading Time: {avg_model_loading_large:.3f} ms")
    print(f"First Inference Time: {avg_first_inference_large:.3f} ms")
    print(f"Total Cold Start Time: {avg_total_cold_start_large:.3f} ms")
    print(f"Start RSS: {avg_start_rss_large:.2f} MB")
    print(f"Peak RSS: {avg_peak_rss_large:.2f} MB")

    print("\n===== Small Model (YOLO11n) 20-Run Average =====")
    print(f"Session Creation Time: {avg_session_creation_small:.3f} ms")
    print(f"Model Loading Time: {avg_model_loading_small:.3f} ms")
    print(f"First Inference Time: {avg_first_inference_small:.3f} ms")
    print(f"Total Cold Start Time: {avg_total_cold_start_small:.3f} ms")
    print(f"Start RSS: {avg_start_rss_small:.2f} MB")
    print(f"Peak RSS: {avg_peak_rss_small:.2f} MB")

    # Save results
    result_path = os.path.join(base_path, "results", "python_cold_start_decomposition_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Cold Start Decomposition Test Results =====\n\n")
        
        f.write("===== Large Model (YOLO11x) =====\n")
        for i, r in enumerate(results_large):
            f.write(f"===== Run #{i+1} =====\n")
            f.write(f"Session Creation Time: {r.session_creation_time:.5f} ms\n")
            f.write(f"Model Loading Time: {r.model_loading_time:.5f} ms\n")
            f.write(f"First Inference Time: {r.first_inference_time:.5f} ms\n")
            f.write(f"Total Cold Start Time: {r.total_cold_start_time:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n\n")
        
        f.write("===== Large Model (YOLO11x) 20-Run Average =====\n")
        f.write(f"Session Creation Time: {avg_session_creation_large:.5f} ms\n")
        f.write(f"Model Loading Time: {avg_model_loading_large:.5f} ms\n")
        f.write(f"First Inference Time: {avg_first_inference_large:.5f} ms\n")
        f.write(f"Total Cold Start Time: {avg_total_cold_start_large:.5f} ms\n")
        f.write(f"Start RSS: {avg_start_rss_large:.5f} MB\n")
        f.write(f"Peak RSS: {avg_peak_rss_large:.5f} MB\n\n")
        
        f.write("===== Small Model (YOLO11n) =====\n")
        for i, r in enumerate(results_small):
            f.write(f"===== Run #{i+1} =====\n")
            f.write(f"Session Creation Time: {r.session_creation_time:.5f} ms\n")
            f.write(f"Model Loading Time: {r.model_loading_time:.5f} ms\n")
            f.write(f"First Inference Time: {r.first_inference_time:.5f} ms\n")
            f.write(f"Total Cold Start Time: {r.total_cold_start_time:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n\n")
        
        f.write("===== Small Model (YOLO11n) 20-Run Average =====\n")
        f.write(f"Session Creation Time: {avg_session_creation_small:.5f} ms\n")
        f.write(f"Model Loading Time: {avg_model_loading_small:.5f} ms\n")
        f.write(f"First Inference Time: {avg_first_inference_small:.5f} ms\n")
        f.write(f"Total Cold Start Time: {avg_total_cold_start_small:.5f} ms\n")
        f.write(f"Start RSS: {avg_start_rss_small:.5f} MB\n")
        f.write(f"Peak RSS: {avg_peak_rss_small:.5f} MB\n")

    print(f"\nResults saved to: {result_path}")
    print("Test complete!")

if __name__ == "__main__":
    main()
