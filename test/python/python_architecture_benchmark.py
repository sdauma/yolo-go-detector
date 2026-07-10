# -*- coding: utf-8 -*-
# python_architecture_benchmark.py
# Python Concurrent Inference Architecture Comparison Test (3 architectures: Shared / Mutex / Session Pool)
#
# Technical Notes:
# - Uses Python baseline Session API (InferenceSession)
# - Session Pool is implemented as a string constant in Architecture enum,
#   actual sessions are still created via InferenceSession
# - Concurrency implemented via ThreadPoolExecutor
#
# Test Purpose:
# - Compare performance characteristics of three concurrent inference architectures
# - Align with Go-side architecture comparison test, provide cross-language data

import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
import threading
import enum

class Architecture(enum.Enum):
    SHARED_SESSION = "Shared"
    MUTEX_PROTECTED = "Mutex"
    SESSION_POOL = "SessionPool"

@dataclass
class TestResult:
    architecture: Architecture
    concurrency: int = 0
    pool_size: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    throughput: float = 0.0
    start_rss: float = 0.0
    peak_rss: float = 0.0
    end_rss: float = 0.0
    rss_drift: float = 0.0

def get_process_rss():
    """返回进程私有内存（PrivateMemorySize64），与 Go 端 memutil.PrivateMemoryMB() 对齐"""
    process = psutil.Process(os.getpid())
    return process.memory_info().private / 1024 / 1024

def create_session(model_path: str, intra_op_threads: int = 1) -> ort.InferenceSession:
    """创建 ONNX Runtime Session"""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    return session

def run_inference(session: ort.InferenceSession, input_data: np.ndarray) -> tuple:
    """执行单次推理"""
    try:
        input_name = session.get_inputs()[0].name
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        latency = (time.perf_counter() - start_time) * 1000
        return latency, None
    except Exception as e:
        return 0.0, str(e)

def calculate_percentiles(latencies: List[float]) -> tuple:
    """计算百分位数"""
    if len(latencies) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sorted_latencies = sorted(latencies)

    min_lat = sorted_latencies[0]
    max_lat = sorted_latencies[-1]
    avg_lat = sum(sorted_latencies) / len(sorted_latencies)

    p50_index = int(len(sorted_latencies) * 0.5)
    p90_index = int(len(sorted_latencies) * 0.9)
    p99_index = int(len(sorted_latencies) * 0.99)

    p50 = sorted_latencies[min(p50_index, len(sorted_latencies) - 1)]
    p90 = sorted_latencies[min(p90_index, len(sorted_latencies) - 1)]
    p99 = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]

    return min_lat, max_lat, avg_lat, p90, p99

def test_shared_session(
    model_path: str,
    input_data: np.ndarray,
    concurrency: int,
    num_requests: int
) -> TestResult:
    """Test Shared Session architecture: multiple threads share a single Session"""
    print(f"Testing Shared Session: {concurrency} concurrency, {num_requests} requests")

    start_rss = get_process_rss()
    peak_rss = start_rss
    peak_rss_lock = threading.Lock()

    session = create_session(model_path, intra_op_threads=1)
    input_name = session.get_inputs()[0].name

    all_latencies = []
    all_errors = []

    def worker(worker_id: int, batch_size: int):
        nonlocal peak_rss
        latencies = []
        errors = []

        for _ in range(batch_size):
            current_rss = get_process_rss()
            with peak_rss_lock:
                peak_rss = max(peak_rss, current_rss)

            # 每次推理创建新的 Tensor 副本（避免 CPU cache 优化）
            input_copy = input_data.copy()

            try:
                start_time = time.perf_counter()
                outputs = session.run(None, {input_name: input_copy})
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))

        return latencies, errors

    start_time = time.perf_counter()
    batch_size = num_requests // concurrency
    remainder = num_requests % concurrency

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(concurrency):
            extra = 1 if i < remainder else 0
            futures.append(executor.submit(worker, i, batch_size + extra))

        for future in as_completed(futures):
            latencies, errors = future.result()
            all_latencies.extend(latencies)
            all_errors.extend(errors)

    total_time = (time.perf_counter() - start_time) * 1000
    end_rss = get_process_rss()

    min_lat, max_lat, avg_lat, p90, p99 = calculate_percentiles(all_latencies)
    throughput = len(all_latencies) / (total_time / 1000.0) if total_time > 0 else 0.0

    return TestResult(
        architecture=Architecture.SHARED_SESSION,
        concurrency=concurrency,
        total_requests=num_requests,
        successful_requests=len(all_latencies),
        failed_requests=len(all_errors),
        total_time=total_time,
        min_latency=min_lat,
        max_latency=max_lat,
        avg_latency=avg_lat,
        p90_latency=p90,
        p99_latency=p99,
        throughput=throughput,
        start_rss=start_rss,
        peak_rss=peak_rss,
        end_rss=end_rss,
        rss_drift=end_rss - start_rss
    )

def test_mutex_protected(
    model_path: str,
    input_data: np.ndarray,
    concurrency: int,
    num_requests: int
) -> TestResult:
    """Test Mutex Protected architecture: serialized Session access"""
    print(f"Testing Mutex Protected: {concurrency} concurrency, {num_requests} requests")

    start_rss = get_process_rss()
    peak_rss = start_rss
    peak_rss_lock = threading.Lock()

    session = create_session(model_path, intra_op_threads=1)
    input_name = session.get_inputs()[0].name
    mutex = threading.Lock()

    all_latencies = []
    all_errors = []

    def worker(worker_id: int, batch_size: int):
        nonlocal peak_rss
        latencies = []
        errors = []

        for _ in range(batch_size):
            current_rss = get_process_rss()
            with peak_rss_lock:
                peak_rss = max(peak_rss, current_rss)

            # 每次推理创建新的 Tensor 副本（避免 CPU cache 优化）
            input_copy = input_data.copy()

            with mutex:
                try:
                    start_time = time.perf_counter()
                    outputs = session.run(None, {input_name: input_copy})
                    latency = (time.perf_counter() - start_time) * 1000
                    latencies.append(latency)
                except Exception as e:
                    errors.append(str(e))

        return latencies, errors

    start_time = time.perf_counter()
    batch_size = num_requests // concurrency
    remainder = num_requests % concurrency

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(concurrency):
            extra = 1 if i < remainder else 0
            futures.append(executor.submit(worker, i, batch_size + extra))

        for future in as_completed(futures):
            latencies, errors = future.result()
            all_latencies.extend(latencies)
            all_errors.extend(errors)

    total_time = (time.perf_counter() - start_time) * 1000
    end_rss = get_process_rss()

    min_lat, max_lat, avg_lat, p90, p99 = calculate_percentiles(all_latencies)
    throughput = len(all_latencies) / (total_time / 1000.0) if total_time > 0 else 0.0

    return TestResult(
        architecture=Architecture.MUTEX_PROTECTED,
        concurrency=concurrency,
        total_requests=num_requests,
        successful_requests=len(all_latencies),
        failed_requests=len(all_errors),
        total_time=total_time,
        min_latency=min_lat,
        max_latency=max_lat,
        avg_latency=avg_lat,
        p90_latency=p90,
        p99_latency=p99,
        throughput=throughput,
        start_rss=start_rss,
        peak_rss=peak_rss,
        end_rss=end_rss,
        rss_drift=end_rss - start_rss
    )

def test_session_pool(
    model_path: str,
    input_data: np.ndarray,
    pool_size: int,
    num_requests: int
) -> TestResult:
    """Test Session Pool architecture: multi-Session pooling"""
    print(f"Testing Session Pool: pool_size={pool_size}, {num_requests} requests")

    start_rss = get_process_rss()
    peak_rss = start_rss
    peak_rss_lock = threading.Lock()

    session_pool = []
    for i in range(pool_size):
        session = create_session(model_path, intra_op_threads=1)
        session_pool.append(session)

    session_queue = []
    for session in session_pool:
        session_queue.append(session)
    queue_lock = threading.Lock()

    all_latencies = []
    all_errors = []

    def worker(worker_id: int, batch_size: int):
        nonlocal peak_rss
        latencies = []
        errors = []

        for _ in range(batch_size):
            current_rss = get_process_rss()
            with peak_rss_lock:
                peak_rss = max(peak_rss, current_rss)

            with queue_lock:
                session = session_queue.pop(0)

            try:
                input_name = session.get_inputs()[0].name
                # 每次推理创建新的 Tensor 副本（避免 CPU cache 优化）
                input_copy = input_data.copy()
                start_time = time.perf_counter()
                outputs = session.run(None, {input_name: input_copy})
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))
            finally:
                with queue_lock:
                    session_queue.append(session)

        return latencies, errors

    start_time = time.perf_counter()
    batch_size = num_requests // pool_size
    remainder = num_requests % pool_size

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = []
        for i in range(pool_size):
            extra = 1 if i < remainder else 0
            futures.append(executor.submit(worker, i, batch_size + extra))

        for future in as_completed(futures):
            latencies, errors = future.result()
            all_latencies.extend(latencies)
            all_errors.extend(errors)

    total_time = (time.perf_counter() - start_time) * 1000
    end_rss = get_process_rss()

    min_lat, max_lat, avg_lat, p90, p99 = calculate_percentiles(all_latencies)
    throughput = len(all_latencies) / (total_time / 1000.0) if total_time > 0 else 0.0

    return TestResult(
        architecture=Architecture.SESSION_POOL,
        pool_size=pool_size,
        total_requests=num_requests,
        successful_requests=len(all_latencies),
        failed_requests=len(all_errors),
        total_time=total_time,
        min_latency=min_lat,
        max_latency=max_lat,
        avg_latency=avg_lat,
        p90_latency=p90,
        p99_latency=p99,
        throughput=throughput,
        start_rss=start_rss,
        peak_rss=peak_rss,
        end_rss=end_rss,
        rss_drift=end_rss - start_rss
    )

def main():
    print("===== Python Inference Architecture Performance Comparison (Paper Level) =====")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(base_path, "third_party", "yolo11x.onnx")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")

    input_data = np.fromfile(input_data_path, dtype=np.float32)
    input_data = input_data.reshape(1, 3, 640, 640)

    all_results = []

    print("\n===== Experiment 1: Shared Session Scalability Test =====")
    for concurrency in [1, 2, 4, 8, 12]:
        result = test_shared_session(model_path, input_data, concurrency, 500)
        all_results.append(result)
        # Console output keeps 2 decimal places (for readability), file saves keep 5 decimal places
        print(f"concurrency={concurrency}, throughput={result.throughput:.2f} REQ/s, avg_latency={result.avg_latency:.2f} ms")

    print("\n===== Experiment 2: Mutex Protected Serialization Test =====")
    for concurrency in [1, 2, 4, 8, 12]:
        result = test_mutex_protected(model_path, input_data, concurrency, 500)
        all_results.append(result)
        # Console output keeps 2 decimal places (for readability), file saves keep 5 decimal places
        print(f"concurrency={concurrency}, throughput={result.throughput:.2f} REQ/s, avg_latency={result.avg_latency:.2f} ms")

    print("\n===== Experiment 3: Session Pool Size Optimization Test =====")
    for pool_size in [1, 2, 4, 6, 8, 12]:
        result = test_session_pool(model_path, input_data, pool_size, 500)
        all_results.append(result)
        # Console output keeps 2 decimal places (for readability), file saves keep 5 decimal places
        print(f"pool_size={pool_size}, throughput={result.throughput:.2f} REQ/s, avg_latency={result.avg_latency:.2f} ms")

    result_path = os.path.join(base_path, "results", "python_architecture_comparison.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Intermediate data keeps 5 decimal places, conforming to core journal standards
    content = "===== Python Inference Architecture Performance Comparison Results =====\n\n"
    for r in all_results:
        config = f"concurrency={r.concurrency}" if r.architecture != Architecture.SESSION_POOL else f"pool_size={r.pool_size}"
        content += f"architecture={r.architecture.value}, {config}, "
        content += f"throughput={r.throughput:.5f} REQ/s, avg_latency={r.avg_latency:.5f} ms, "
        content += f"P50={r.p50_latency:.5f} ms, P90={r.p90_latency:.5f} ms, P99={r.p99_latency:.5f} ms, "
        content += f"min_latency={r.min_latency:.5f} ms, max_latency={r.max_latency:.5f} ms, "
        content += f"peak_rss={r.peak_rss:.5f} MB, rss_drift={r.rss_drift:.5f} MB\n"

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nResults saved to: {result_path}")
    print("\n===== Experiment Completed =====")

if __name__ == "__main__":
    main()
