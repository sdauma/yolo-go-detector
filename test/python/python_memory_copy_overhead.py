import os
import time
import numpy as np
import onnxruntime as ort
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

@dataclass
class MemoryCopyResult:
    data_copy_time: float
    c_call_time: float
    gc_pause_time: float
    total_overhead: float
    inference_time: float
    overhead_percent: float

def get_process_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_data_copy_overhead(input_data: np.ndarray) -> tuple:
    start_time = time.perf_counter()
    
    input_tensor = input_data.copy()
    
    data_copy_time = (time.perf_counter() - start_time) * 1000
    
    c_call_start = time.perf_counter()
    for _ in range(100):
        input_tensor_copy = input_data.copy()
    c_call_time = ((time.perf_counter() - c_call_start) * 1000) / 100.0
    
    return data_copy_time, c_call_time

def measure_gc_pause():
    import gc
    gc.collect()
    start_time = time.perf_counter()
    gc.collect()
    gc_pause_time = (time.perf_counter() - start_time) * 1000
    return gc_pause_time

def run_memory_copy_benchmark(session: ort.InferenceSession, input_data: np.ndarray, input_name: str) -> MemoryCopyResult:
    data_copy_time, c_call_time = measure_data_copy_overhead(input_data)
    gc_pause_time = measure_gc_pause()
    
    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: input_data})
    inference_time = (time.perf_counter() - start_time) * 1000
    
    total_overhead = data_copy_time + c_call_time + gc_pause_time
    overhead_percent = (total_overhead / inference_time) * 100
    
    return MemoryCopyResult(
        data_copy_time=data_copy_time,
        c_call_time=c_call_time,
        gc_pause_time=gc_pause_time,
        total_overhead=total_overhead,
        inference_time=inference_time,
        overhead_percent=overhead_percent
    )

def main():
    print("===== Python 内存拷贝和线程调度开销测试 =====")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(base_path, "third_party", "yolo11x.onnx")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    
    input_data = np.fromfile(input_data_path, dtype=np.float32)
    input_data = input_data.reshape(1, 3, 640, 640)
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 12
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    print("\n===== 内存拷贝开销分析 =====")
    results = []
    
    for i in range(10):
        result = run_memory_copy_benchmark(session, input_data, input_name)
        results.append(result)
    
    avg_data_copy = sum(r.data_copy_time for r in results) / len(results)
    avg_c_call = sum(r.c_call_time for r in results) / len(results)
    avg_gc_pause = sum(r.gc_pause_time for r in results) / len(results)
    avg_total_overhead = sum(r.total_overhead for r in results) / len(results)
    avg_inference = sum(r.inference_time for r in results) / len(results)
    avg_overhead_percent = sum(r.overhead_percent for r in results) / len(results)
    
    print(f"数据拷贝时间: {avg_data_copy:.5f} ms")
    print(f"C调用开销: {avg_c_call:.5f} ms")
    print(f"GC暂停时间: {avg_gc_pause:.5f} ms")
    print(f"总开销时间: {avg_total_overhead:.5f} ms")
    print(f"推理时间: {avg_inference:.5f} ms")
    print(f"开销占比: {avg_overhead_percent:.5f}%")
    
    print("\n===== 线程调度开销测试 =====")
    thread_counts = [1, 2, 4, 8, 12]
    
    for thread_count in thread_counts:
        print(f"\n测试 {thread_count} 线程配置...")
        
        times = []
        lock = threading.Lock()
        
        def measure_thread_switch():
            start_time = time.perf_counter()
            time.sleep(0)
            elapsed = (time.perf_counter() - start_time) * 1000
            with lock:
                times.append(elapsed)
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for _ in range(100):
                executor.submit(measure_thread_switch)
        
        avg_time = sum(times) / len(times)
        print(f"平均线程调度时间: {avg_time:.5f} ms")
    
    print("\n===== 内存使用分析 =====")
    start_rss = get_process_rss()
    
    for _ in range(100):
        input_tensor = input_data.copy()
        del input_tensor
        import gc
        gc.collect()
    
    end_rss = get_process_rss()
    rss_drift = end_rss - start_rss
    
    print(f"初始RSS: {start_rss:.5f} MB")
    print(f"最终RSS: {end_rss:.5f} MB")
    print(f"RSS漂移: {rss_drift:.5f} MB")
    
    result_path = os.path.join(base_path, "..", "..", "results", "python_memory_copy_overhead_result.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    result_content = "===== Python 内存拷贝和线程调度开销测试结果 =====\n\n"
    result_content += f"数据拷贝时间: {avg_data_copy:.5f} ms\n"
    result_content += f"C调用开销: {avg_c_call:.5f} ms\n"
    result_content += f"GC暂停时间: {avg_gc_pause:.5f} ms\n"
    result_content += f"总开销时间: {avg_total_overhead:.5f} ms\n"
    result_content += f"推理时间: {avg_inference:.5f} ms\n"
    result_content += f"开销占比: {avg_overhead_percent:.5f}%\n\n"
    result_content += f"初始RSS: {start_rss:.5f} MB\n"
    result_content += f"最终RSS: {end_rss:.5f} MB\n"
    result_content += f"RSS漂移: {rss_drift:.5f} MB\n"
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_content)
    
    print(f"\n结果已保存到: {result_path}")

if __name__ == "__main__":
    main()