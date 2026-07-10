# -*- coding: utf-8 -*-
# python_memory_standardization.py
# Python memory standardization test
#
# Technical notes:
# - Uses Python baseline Session API (InferenceSession)
# - Explicitly configures thread params via SessionOptions (intraOp=12, interOp=1)
# - Uses sess.run() standard call path, no I/O Binding
# - Binds to first 4 CPU cores (cpu_affinity=[0,1,2,3])
# - Runs 10 inferences to stabilize memory before sampling
#
# Test purpose:
# - Record interpreter resident memory (base memory)
# - Record memory after model loading
# - Record memory after inference
# - Run multiple tests, calculate averages
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
class MemoryResult:
    interpreter_memory: float  # Interpreter resident memory
    model_loaded_memory: float  # Memory after model loading
    post_inference_memory: float  # Memory after inference
    memory_increase: float  # Memory increase (model loading + inference)

def run_memory_test(model_path, model_name):
    print(f"\n===== Python Memory Test - {model_name} ====")
    
    # Bind CPU cores
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1, 2, 3])
    
    # 1. Measure interpreter resident memory (base memory)
    print("Measuring interpreter resident memory...")
    interpreter_memory = process.memory_info().private / 1024 / 1024
    print(f"Interpreter Resident Memory: {interpreter_memory:.2f} MB")
    
    # 2. Create Session (load model)
    print("Loading model...")
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
    
    # 3. Measure memory after model loading
    model_loaded_memory = process.memory_info().private / 1024 / 1024
    print(f"Memory After Model Loading: {model_loaded_memory:.2f} MB")
    
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

    # 4. Run inference
    print("Running inference...")
    for _ in range(10):  # Run 10 inferences to stabilize memory usage
        sess.run(None, {input_name: input_data})
    
    # 5. Measure memory after inference
    post_inference_memory = process.memory_info().private / 1024 / 1024
    print(f"Memory After Inference: {post_inference_memory:.2f} MB")
    
    # Calculate memory increase
    memory_increase = post_inference_memory - interpreter_memory
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    return MemoryResult(
        interpreter_memory=interpreter_memory,
        model_loaded_memory=model_loaded_memory,
        post_inference_memory=post_inference_memory,
        memory_increase=memory_increase
    )

def main():
    print("===== Python Memory Standardization Test (10 runs) =====")

    # Run 10 tests - large model
    num_runs = 10
    results_large = []
    results_small = []

    print("\n===== Testing Large Model (YOLO11x) =====")
    for i in range(num_runs):
        print(f"\n===== Run #{i+1} =====")
        result = run_memory_test(model_path_large, "YOLO11x")
        results_large.append(result)

    print("\n===== Testing Small Model (YOLO11n) =====")
    for i in range(num_runs):
        print(f"\n===== Run #{i+1} =====")
        result = run_memory_test(model_path_small, "YOLO11n")
        results_small.append(result)

    # Calculate large model averages
    avg_interpreter_large = sum(r.interpreter_memory for r in results_large) / num_runs
    avg_model_loaded_large = sum(r.model_loaded_memory for r in results_large) / num_runs
    avg_post_inference_large = sum(r.post_inference_memory for r in results_large) / num_runs
    avg_memory_increase_large = sum(r.memory_increase for r in results_large) / num_runs

    # Calculate small model averages
    avg_interpreter_small = sum(r.interpreter_memory for r in results_small) / num_runs
    avg_model_loaded_small = sum(r.model_loaded_memory for r in results_small) / num_runs
    avg_post_inference_small = sum(r.post_inference_memory for r in results_small) / num_runs
    avg_memory_increase_small = sum(r.memory_increase for r in results_small) / num_runs

    print("\n===== Large Model (YOLO11x) 10-Run Average =====")
    print(f"Interpreter Resident Memory: {avg_interpreter_large:.2f} MB")
    print(f"Memory After Model Loading: {avg_model_loaded_large:.2f} MB")
    print(f"Memory After Inference: {avg_post_inference_large:.2f} MB")
    print(f"Memory Increase: {avg_memory_increase_large:.2f} MB")

    print("\n===== Small Model (YOLO11n) 10-Run Average =====")
    print(f"Interpreter Resident Memory: {avg_interpreter_small:.2f} MB")
    print(f"Memory After Model Loading: {avg_model_loaded_small:.2f} MB")
    print(f"Memory After Inference: {avg_post_inference_small:.2f} MB")
    print(f"Memory Increase: {avg_memory_increase_small:.2f} MB")

    # Save results
    result_path = os.path.join(base_path, "results", "python_memory_standardization_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Memory Standardization Test Results =====\n\n")
        
        f.write("===== Large Model (YOLO11x) =====\n")
        for i, r in enumerate(results_large):
            f.write(f"===== Run #{i+1} =====\n")
            f.write(f"Interpreter Resident Memory: {r.interpreter_memory:.5f} MB\n")
            f.write(f"Memory After Model Loading: {r.model_loaded_memory:.5f} MB\n")
            f.write(f"Memory After Inference: {r.post_inference_memory:.5f} MB\n")
            f.write(f"Memory Increase: {r.memory_increase:.5f} MB\n\n")
        
        f.write("===== Large Model (YOLO11x) 10-Run Average =====\n")
        f.write(f"Interpreter Resident Memory: {avg_interpreter_large:.5f} MB\n")
        f.write(f"Memory After Model Loading: {avg_model_loaded_large:.5f} MB\n")
        f.write(f"Memory After Inference: {avg_post_inference_large:.5f} MB\n")
        f.write(f"Memory Increase: {avg_memory_increase_large:.5f} MB\n\n")
        
        f.write("===== Small Model (YOLO11n) =====\n")
        for i, r in enumerate(results_small):
            f.write(f"===== Run #{i+1} =====\n")
            f.write(f"Interpreter Resident Memory: {r.interpreter_memory:.5f} MB\n")
            f.write(f"Memory After Model Loading: {r.model_loaded_memory:.5f} MB\n")
            f.write(f"Memory After Inference: {r.post_inference_memory:.5f} MB\n")
            f.write(f"Memory Increase: {r.memory_increase:.5f} MB\n\n")
        
        f.write("===== Small Model (YOLO11n) 10-Run Average =====\n")
        f.write(f"Interpreter Resident Memory: {avg_interpreter_small:.5f} MB\n")
        f.write(f"Memory After Model Loading: {avg_model_loaded_small:.5f} MB\n")
        f.write(f"Memory After Inference: {avg_post_inference_small:.5f} MB\n")
        f.write(f"Memory Increase: {avg_memory_increase_small:.5f} MB\n")

    print(f"\nResults saved to: {result_path}")
    print("Test complete!")

if __name__ == "__main__":
    main()
