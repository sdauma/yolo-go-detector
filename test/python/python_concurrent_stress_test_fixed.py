import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List
import threading

@dataclass
class ConcurrentTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    throughput: float
    start_rss: float
    peak_rss: float
    end_rss: float
    rss_drift: float

def get_process_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_concurrent_worker(
    worker_id: int,
    model_path: str,
    input_data: np.ndarray,
    batch_size: int,
    peak_rss_list: List[float],
    peak_rss_lock: threading.Lock
) -> tuple:
    """
    每个worker创建独立的Session，避免线程安全问题
    """
    try:
        # 每个worker独立创建Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # 并发时用单线程
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        input_name = session.get_inputs()[0].name
        
        latencies = []
        errors = []
        
        for _ in range(batch_size):
            # 更新峰值RSS
            current_rss = get_process_rss()
            with peak_rss_lock:
                peak_rss_list[0] = max(peak_rss_list[0], current_rss)
            
            try:
                start_time = time.perf_counter()
                outputs = session.run(None, {input_name: input_data})
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))
        
        return latencies, errors
        
    except Exception as e:
        return [], [str(e)]

def calculate_percentiles(latencies: List[float]) -> tuple:
    if len(latencies) == 0:
        return 0.0, 0.0, 0.0
    
    sorted_latencies = sorted(latencies)
    
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
    p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    # 处理边界情况
    p50 = p50 if len(sorted_latencies) > 0 else 0.0
    p90_index = int(len(sorted_latencies) * 0.9)
    p99_index = int(len(sorted_latencies) * 0.99)
    p90 = sorted_latencies[min(p90_index, len(sorted_latencies) - 1)]
    p99 = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
    
    return p50, p90, p99

def run_concurrent_test(
    model_path: str,
    input_data: np.ndarray,
    concurrency: int,
    num_requests: int
) -> ConcurrentTestResult:
    print(f"执行并发测试: {concurrency} 并发, {num_requests} 请求, intra_op_threads=1")
    
    start_rss = get_process_rss()
    peak_rss_list = [start_rss]  # 使用list实现可变引用
    peak_rss_lock = threading.Lock()
    
    start_time = time.perf_counter()
    
    batch_size = num_requests // concurrency
    
    all_latencies = []
    all_errors = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        
        for i in range(concurrency):
            future = executor.submit(
                run_concurrent_worker,
                i,
                model_path,
                input_data,
                batch_size,
                peak_rss_list,
                peak_rss_lock
            )
            futures.append(future)
        
        for future in as_completed(futures):
            latencies, errors = future.result()
            all_latencies.extend(latencies)
            all_errors.extend(errors)
    
    total_time = (time.perf_counter() - start_time) * 1000
    end_rss = get_process_rss()
    peak_rss = peak_rss_list[0]
    
    if len(all_latencies) == 0:
        return ConcurrentTestResult(
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=num_requests,
            total_time=total_time,
            avg_latency=0.0,
            p50_latency=0.0,
            p90_latency=0.0,
            p99_latency=0.0,
            min_latency=0.0,
            max_latency=0.0,
            throughput=0.0,
            start_rss=start_rss,
            peak_rss=peak_rss,
            end_rss=end_rss,
            rss_drift=end_rss - start_rss
        )
    
    sum_latency = sum(all_latencies)
    avg_latency = sum_latency / len(all_latencies)
    min_latency = min(all_latencies)
    max_latency = max(all_latencies)
    p50, p90, p99 = calculate_percentiles(all_latencies)
    
    throughput = len(all_latencies) / (total_time / 1000.0)
    
    return ConcurrentTestResult(
        total_requests=num_requests,
        successful_requests=len(all_latencies),
        failed_requests=len(all_errors),
        total_time=total_time,
        avg_latency=avg_latency,
        p50_latency=p50,
        p90_latency=p90,
        p99_latency=p99,
        min_latency=min_latency,
        max_latency=max_latency,
        throughput=throughput,
        start_rss=start_rss,
        peak_rss=peak_rss,
        end_rss=end_rss,
        rss_drift=end_rss - start_rss
    )

def main():
    print("===== Python 并发推理性能测试（学术版）=====")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(base_path, "third_party", "yolo11x.onnx")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    
    input_data = np.fromfile(input_data_path, dtype=np.float32)
    input_data = input_data.reshape(1, 3, 640, 640)
    
    print("\n===== Session Pool 扩展性测试（并发度 vs CPU 核心数）=====")
    concurrency_levels = [1, 2, 4, 6, 8, 12]
    num_requests = 500
    
    results = []
    
    for concurrency in concurrency_levels:
        print(f"\n===== 测试配置: {concurrency} 并发 =====")
        result = run_concurrent_test(model_path, input_data, concurrency, num_requests)
        results.append(result)
        
        print(f"总请求数: {result.total_requests}")
        print(f"成功请求数: {result.successful_requests}")
        print(f"失败请求数: {result.failed_requests}")
        print(f"总时间: {result.total_time:.2f} ms")
        print(f"平均延迟: {result.avg_latency:.2f} ms")
        print(f"P50延迟: {result.p50_latency:.2f} ms")
        print(f"P90延迟: {result.p90_latency:.2f} ms")
        print(f"P99延迟: {result.p99_latency:.2f} ms")
        print(f"最小延迟: {result.min_latency:.2f} ms")
        print(f"最大延迟: {result.max_latency:.2f} ms")
        print(f"吞吐量: {result.throughput:.2f} REQ/s")
        print(f"初始RSS: {result.start_rss:.2f} MB")
        print(f"峰值RSS: {result.peak_rss:.2f} MB")
        print(f"最终RSS: {result.end_rss:.2f} MB")
        print(f"RSS漂移: {result.rss_drift:.2f} MB")
    
    result_path = os.path.join(base_path, "results", "python_session_pool_performance.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    result_content = "===== Python Session Pool 并发推理性能测试结果（学术版）=====\n\n"
    # 中间数据保留5位小数，符合核心期刊规范
    for i, result in enumerate(results):
        concurrency = concurrency_levels[i]
        result_content += f"===== 测试配置: {concurrency} 并发 =====\n"
        result_content += f"总请求数: {result.total_requests}\n"
        result_content += f"成功请求数: {result.successful_requests}\n"
        result_content += f"失败请求数: {result.failed_requests}\n"
        result_content += f"总时间: {result.total_time:.5f} ms\n"
        result_content += f"平均延迟: {result.avg_latency:.5f} ms\n"
        result_content += f"P50延迟: {result.p50_latency:.5f} ms\n"
        result_content += f"P90延迟: {result.p90_latency:.5f} ms\n"
        result_content += f"P99延迟: {result.p99_latency:.5f} ms\n"
        result_content += f"最小延迟: {result.min_latency:.5f} ms\n"
        result_content += f"最大延迟: {result.max_latency:.5f} ms\n"
        result_content += f"吞吐量: {result.throughput:.5f} REQ/s\n"
        result_content += f"初始RSS: {result.start_rss:.5f} MB\n"
        result_content += f"峰值RSS: {result.peak_rss:.5f} MB\n"
        result_content += f"最终RSS: {result.end_rss:.5f} MB\n"
        result_content += f"RSS漂移: {result.rss_drift:.5f} MB\n\n"
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_content)
    
    print(f"\n结果已保存到: {result_path}")
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    main()
