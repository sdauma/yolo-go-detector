# -*- coding: utf-8 -*-
# python_memory_copy_overhead.py
# Python 鍐呭瓨鎷疯礉寮€閿€鍒嗘瀽娴嬭瘯
#
# 鎶€鏈鏄庯細
# - 浣跨敤 Python baseline Session 鎺ュ彛锛圛nferenceSession锛?
# - 閫氳繃 SessionOptions 鏄惧紡閰嶇疆threads鍙傛暟
#
# 娴嬭瘯鐩殑锛?
# - 娴嬮噺鎺ㄧ悊杩囩▼涓?Data Copy銆丆 Call銆丟C Pause 绛夌幆鑺傜殑鏃堕棿鍗犳瘮
# - 涓?Go 绔唴瀛樻嫹璐濆紑閿€娴嬭瘯瀵归綈

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
    """杩斿洖杩涚▼绉佹湁鍐呭瓨锛圥rivateMemorySize64锛夛紝涓?Go 绔?memutil.PrivateMemoryMB() 瀵归綈"""
    process = psutil.Process(os.getpid())
    return process.memory_info().private / 1024 / 1024

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
    print("===== Python Memory Copy and Thread Scheduling Overhead Test =====")
    
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
    
    print("\n===== Memory Copy Overhead Analysis =====")
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
    
    print(f"鏁版嵁鎷疯礉鏃堕棿: {avg_data_copy:.5f} ms")
    print(f"C璋冪敤寮€閿€: {avg_c_call:.5f} ms")
    print(f"GC鏆傚仠鏃堕棿: {avg_gc_pause:.5f} ms")
    print(f"鎬诲紑閿€鏃堕棿: {avg_total_overhead:.5f} ms")
    print(f"鎺ㄧ悊鏃堕棿: {avg_inference:.5f} ms")
    print(f"寮€閿€鍗犳瘮: {avg_overhead_percent:.5f}%")
    
    print("\n===== Thread Scheduling Overhead Test =====")
    thread_counts = [1, 2, 4, 8, 12]
    
    for thread_count in thread_counts:
        print(f"\n娴嬭瘯 {thread_count} threads閰嶇疆...")
        
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
        print(f"骞冲潎threads璋冨害鏃堕棿: {avg_time:.5f} ms")
    
    print("\n===== Memory Usage Analysis =====")
    start_rss = get_process_rss()
    
    for _ in range(100):
        input_tensor = input_data.copy()
        del input_tensor
        import gc
        gc.collect()
    
    end_rss = get_process_rss()
    rss_drift = end_rss - start_rss
    
    print(f"start_rss: {start_rss:.5f} MB")
    print(f"end_rss: {end_rss:.5f} MB")
    print(f"rss_drift: {rss_drift:.5f} MB")
    
    result_path = os.path.join(base_path, "..", "..", "results", "python_memory_copy_overhead_result.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    result_content = "===== Python 鍐呭瓨鎷疯礉鍜宼hreads璋冨害寮€閿€娴嬭瘯缁撴灉 =====\n\n"
    result_content += f"鏁版嵁鎷疯礉鏃堕棿: {avg_data_copy:.5f} ms\n"
    result_content += f"C璋冪敤寮€閿€: {avg_c_call:.5f} ms\n"
    result_content += f"GC鏆傚仠鏃堕棿: {avg_gc_pause:.5f} ms\n"
    result_content += f"鎬诲紑閿€鏃堕棿: {avg_total_overhead:.5f} ms\n"
    result_content += f"鎺ㄧ悊鏃堕棿: {avg_inference:.5f} ms\n"
    result_content += f"寮€閿€鍗犳瘮: {avg_overhead_percent:.5f}%\n\n"
    result_content += f"start_rss: {start_rss:.5f} MB\n"
    result_content += f"end_rss: {end_rss:.5f} MB\n"
    result_content += f"rss_drift: {rss_drift:.5f} MB\n"
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_content)
    
    print(f"\nResults saved to: {result_path}")

if __name__ == "__main__":
    main()
