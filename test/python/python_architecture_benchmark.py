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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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
    """测试 Shared Session 架构：多线程共享单一 Session"""
    print(f"测试 Shared Session: {concurrency} 并发，{num_requests} 请求")

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

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i, batch_size) for i in range(concurrency)]

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
    """测试 Mutex Protected 架构：串行访问 Session"""
    print(f"测试 Mutex Protected: {concurrency} 并发，{num_requests} 请求")

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

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i, batch_size) for i in range(concurrency)]

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
    """测试 Session Pool 架构：多 Session 池化"""
    print(f"测试 Session Pool: pool_size={pool_size}, {num_requests} 请求")

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

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(worker, i, batch_size) for i in range(pool_size)]

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
    print("===== Python 推理架构性能对比实验（论文级）=====")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(base_path, "third_party", "yolo11x.onnx")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")

    input_data = np.fromfile(input_data_path, dtype=np.float32)
    input_data = input_data.reshape(1, 3, 640, 640)

    all_results = []

    print("\n===== 实验 1: Shared Session 扩展性测试 =====")
    for concurrency in [1, 2, 4, 8, 12]:
        result = test_shared_session(model_path, input_data, concurrency, 500)
        all_results.append(result)
        # 控制台输出保留2位小数（便于阅读），文件保存保留5位小数
        print(f"并发={concurrency}, 吞吐量={result.throughput:.2f} REQ/s, 平均延迟={result.avg_latency:.2f} ms")

    print("\n===== 实验 2: Mutex Protected 串行化测试 =====")
    for concurrency in [1, 2, 4, 8, 12]:
        result = test_mutex_protected(model_path, input_data, concurrency, 500)
        all_results.append(result)
        # 控制台输出保留2位小数（便于阅读），文件保存保留5位小数
        print(f"并发={concurrency}, 吞吐量={result.throughput:.2f} REQ/s, 平均延迟={result.avg_latency:.2f} ms")

    print("\n===== 实验 3: Session Pool 池大小优化测试 =====")
    for pool_size in [1, 2, 4, 6, 8, 12]:
        result = test_session_pool(model_path, input_data, pool_size, 500)
        all_results.append(result)
        # 控制台输出保留2位小数（便于阅读），文件保存保留5位小数
        print(f"池大小={pool_size}, 吞吐量={result.throughput:.2f} REQ/s, 平均延迟={result.avg_latency:.2f} ms")

    result_path = os.path.join(base_path, "results", "python_architecture_comparison.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # 中间数据保留5位小数，符合核心期刊规范
    content = "===== Python 推理架构性能对比实验结果 =====\n\n"
    for r in all_results:
        config = f"并发={r.concurrency}" if r.architecture != Architecture.SESSION_POOL else f"池大小={r.pool_size}"
        content += f"架构={r.architecture.value}, {config}, "
        content += f"吞吐量={r.throughput:.5f} REQ/s, 平均延迟={r.avg_latency:.5f} ms, "
        content += f"P50={r.p50_latency:.5f} ms, P90={r.p90_latency:.5f} ms, P99={r.p99_latency:.5f} ms, "
        content += f"最小延迟={r.min_latency:.5f} ms, 最大延迟={r.max_latency:.5f} ms, "
        content += f"峰值RSS={r.peak_rss:.5f} MB, RSS漂移={r.rss_drift:.5f} MB\n"

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n结果已保存到：{result_path}")
    print("\n===== 实验完成 =====")

if __name__ == "__main__":
    main()
