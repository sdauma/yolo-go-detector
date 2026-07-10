# -*- coding: utf-8 -*-
# python_session_pool_ablation.py
# Python Session Pool 娑堣瀺瀹為獙锛圓blation Study锛?
#
# 鎶€鏈鏄庯細
# - 浣跨敤 Python baseline Session 鎺ュ彛锛圛nferenceSession锛夛紝閫氳繃 ThreadPoolExecutor 妯℃嫙姹燾oncurrency
# - 閫氳繃 SessionOptions 鏄惧紡閰嶇疆 intra_op_num_threads锛圥2鍘熷垯锛氱姝緷璧栭粯璁ゅ€硷級
# - 娴嬭瘯涓嶅悓 PoolSize 脳 IntraThreads 缁勫悎
#
# P0鍘熷垯锛氭祴璇曚粎鐢ㄤ簬瑙傚療鐜拌薄锛屼笉鐢ㄤ簬璇█绾ф€ц兘缁撹
#
# 娴嬭瘯鐩殑锛?
# - 鍥炵瓟"涓嶅悓pool_size鍜宼hreads閰嶇疆濡備綍褰卞搷throughput銆佸欢杩熷拰鍐呭瓨"
# - 涓鸿鏂囪ˉ鍏呮秷铻嶅疄楠岋紙Ablation Study锛夋暟鎹?

import os
import sys
import time
import json
import gc
import numpy as np
import onnxruntime as ort
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List
import threading

@dataclass
class AblationConfig:
    pool_size: int
    intra_threads: int

@dataclass
class AblationResult:
    config: AblationConfig
    model: str
    status: str = "OK"
    skip_reason: str = ""
    total_requests: int = 0
    successful_requests: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    throughput_reqs: float = 0.0
    start_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    end_rss_mb: float = 0.0
    rss_drift_mb: float = 0.0
    duration_sec: float = 0.0


def get_process_rss():
    """鑾峰彇杩涚▼绉佹湁鍐呭瓨锛圥rivateMemorySize64锛夛紝涓?Go 绔?memutil.PrivateMemoryMB() 瀵归綈"""
    process = psutil.Process(os.getpid())
    return process.memory_info().private / 1024 / 1024


def create_session(model_path: str, intra_op_threads: int = 1) -> ort.InferenceSession:
    """鍒涘缓 ONNX Runtime Session锛圥2鍘熷垯锛氭樉寮忚缃墍鏈夊弬鏁帮級"""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3

    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    return session


def run_ablation_test(
    model_path: str,
    input_data: np.ndarray,
    config: AblationConfig,
    model_name: str
) -> AblationResult:
    """杩愯鍗曠粍娑堣瀺瀹為獙"""
    print(f"\n--- 娑堣瀺瀹為獙: PoolSize={config.pool_size}, IntraThreads={config.intra_threads} ---")

    # 鍒涘缓Session Pool锛圥ython涓娇鐢╰hreads瀹夊叏鐨凷ession鍒楄〃+淇″彿閲忔ā鎷燂級
    pool = []
    pool_lock = threading.Lock()
    pool_semaphore = threading.BoundedSemaphore(config.pool_size)

    try:
        for i in range(config.pool_size):
            session = create_session(model_path, config.intra_threads)
            pool.append(session)
    except Exception as e:
        print(f"  [SKIP] Session creation failed: {e}")
        # Clean up any sessions that were created
        for s in pool:
            del s
        pool.clear()
        gc.collect()
        time.sleep(2)
        # Return a result with status=SKIPPED instead of None
        return AblationResult(
            model=model_name,
            config=config,
            status="SKIPPED",
            skip_reason=f"Session creation failed: {e}",
            throughput_reqs=0, avg_latency_ms=0, p99_latency_ms=0,
            peak_rss_mb=0, rss_drift_mb=0
        )

    # 棰勭儹
    print("  Warmup...", end="", flush=True)
    try:
        input_name = pool[0].get_inputs()[0].name
        for session in pool:
            for _ in range(5):
                session.run(None, {input_name: input_data})
        print(" done")
    except Exception as e:
        print(f" failed: {e}")
        # Cleanup on warmup failure
        for s in pool:
            del s
        pool.clear()
        gc.collect()
        time.sleep(2)
        return AblationResult(
            model=model_name,
            config=config,
            status="SKIPPED",
            skip_reason=f"Warmup failed: {e}",
            throughput_reqs=0, avg_latency_ms=0, p99_latency_ms=0,
            peak_rss_mb=0, rss_drift_mb=0
        )

    # 璁板綍鍒濆status
    start_rss = get_process_rss()
    peak_rss = start_rss
    start_time = time.perf_counter()

    latencies = []
    latencies_lock = threading.Lock()
    peak_rss_lock = threading.Lock()

    num_requests = 500
    concurrency = 12
    completed = [0]  # use list for mutable closure
    oom_occurred = [False]  # flag for OOM in worker threads

    def worker(request_count):
        nonlocal peak_rss
        session_idx = 0

        for _ in range(request_count):
            if oom_occurred[0]:
                return  # Stop early if another thread hit OOM

            pool_semaphore.acquire()
            with pool_lock:
                session = pool[session_idx % len(pool)]
                session_idx += 1

            try:
                infer_start = time.perf_counter()
                session.run(None, {input_name: input_data})
                lat = (time.perf_counter() - infer_start) * 1000.0

                with latencies_lock:
                    latencies.append(lat)
                    completed[0] += 1
                    if completed[0] % 100 == 0:
                        print(f"{completed[0]} ", end="", flush=True)

                current_rss = get_process_rss()
                with peak_rss_lock:
                    if current_rss > peak_rss:
                        peak_rss = current_rss
            except Exception as e:
                oom_occurred[0] = True
                pool_semaphore.release()
                return
            finally:
                if not oom_occurred[0]:
                    try:
                        pool_semaphore.release()
                    except ValueError:
                        pass

    print("  Progress: ", end="", flush=True)
    inference_ok = True
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            requests_per_worker = num_requests // concurrency
            remainder = num_requests % concurrency
            for i in range(concurrency):
                extra = 1 if i < remainder else 0
                futures.append(executor.submit(worker, requests_per_worker + extra))
            for f in futures:
                f.result()
    except Exception as e:
        print(f"\n  [OOM] Inference aborted: {e}")
        inference_ok = False

    end_rss = get_process_rss()
    duration = time.perf_counter() - start_time

    # 閲婃斁Session璧勬簮锛堥槻姝㈠唴瀛樻硠婕忥級
    for session in pool:
        # Python ONNX Runtime娌℃湁鏄惧紡鐨刢lose鏂规硶
        # 鍒犻櫎寮曠敤锛岃GC鍥炴敹
        del session
    pool.clear()
    
    # 寮哄埗GC鍥炴敹骞剁瓑寰?
    gc.collect()
    time.sleep(1)

    # 璁＄畻缁熻閲?
    if not latencies:
        if oom_occurred[0]:
            print("  [OOM] No valid data - memory exhausted during inference")
        else:
            print("No valid inference data")
        return AblationResult(
            model=model_name,
            config=config,
            status="SKIPPED",
            skip_reason="No valid inference data" + (" (OOM)" if oom_occurred[0] else ""),
            throughput_reqs=0, avg_latency_ms=0, p99_latency_ms=0,
            peak_rss_mb=0, rss_drift_mb=0
        )

    latencies.sort()
    n = len(latencies)

    avg = np.mean(latencies)
    std = np.std(latencies, ddof=0)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    min_lat = latencies[0]
    max_lat = latencies[-1]
    throughput = n / duration if duration > 0 else 0

    result = AblationResult(
        config=config,
        status="OK",
        total_requests=n,
        successful_requests=n,
        avg_latency_ms=float(avg),
        p50_latency_ms=float(p50),
        p90_latency_ms=float(p90),
        p99_latency_ms=float(p99),
        min_latency_ms=float(min_lat),
        max_latency_ms=float(max_lat),
        std_latency_ms=float(std),
        throughput_reqs=float(throughput),
        start_rss_mb=float(start_rss),
        peak_rss_mb=float(peak_rss),
        end_rss_mb=float(end_rss),
        rss_drift_mb=float(end_rss - start_rss),
        duration_sec=float(duration),
        model=model_name,
    )

    # 寤惰繜灞曠ず淇濈暀3浣嶅皬鏁帮紝缁熻閲忎繚鐣?浣嶅皬鏁帮紝绗﹀悎鏍稿績鏈熷垔瑙勮寖
    print(f"  throughput: {throughput:.2f} REQ/s")
    print(f"  avg_latency: {avg:.3f} ms, P50: {p50:.3f} ms, P99: {p99:.3f} ms")
    print(f"  寤惰繜鏍囧噯宸? {std:.4f} ms")
    print(f"  RSS: 璧峰 {start_rss:.2f} MB, 宄板€?{peak_rss:.2f} MB, 缁撴潫 {end_rss:.2f} MB, "
          f"婕傜Щ {end_rss - start_rss:.2f} MB")

    return result


def main():
    print("===== Python Session Pool Ablation Experiment =====")
    print("Goal: Evaluate the effect of pool_size and threads configuration on Session Pool performance")
    print("")

    # 璺緞閰嶇疆
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..')
    results_dir = os.path.join(base_path, 'results')

    # 鍥哄畾闅忔満绉嶅瓙
    np.random.seed(12345)

    # 鍔犺浇杈撳叆鏁版嵁
    input_data_path = os.path.join(base_path, 'test', 'data', 'input_data.bin')
    input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(1, 3, 640, 640)

    # 娴嬭瘯閰嶇疆
    pool_sizes = [4, 8, 12, 16]
    intra_thread_configs = [1, 2, 4, 8]
    models = [
        ('YOLO11x', 'yolo11x.onnx'),
        ('YOLO11n', 'yolo11n.onnx'),
    ]

    all_results = []

    for model_name, model_file in models:
        model_path = os.path.join(base_path, 'third_party', model_file)

        print(f"\n========== model: {model_name} ==========")

        for pool_size in pool_sizes:
            for threads in intra_thread_configs:
                # 璺宠繃涓嶅悎鐞嗛厤缃紙threads鏁板ぇ浜巔ool_size鏃犳剰涔夛級
                if threads > pool_size:
                    print(f"  [SKIP] pool_size={pool_size}, threads={threads} (threads > pool_size, meaningless configuration)")
                    all_results.append({
                        'model': model_name,
                        'config': {'pool_size': pool_size, 'intra_threads': threads},
                        'status': 'SKIPPED',
                        'reason': 'threads > pool_size',
                        'throughput_reqs': 0, 'avg_latency_ms': 0,
                        'p99_latency_ms': 0, 'peak_rss_mb': 0, 'rss_drift_mb': 0
                    })
                    continue

                config = AblationConfig(pool_size, threads)
                result = run_ablation_test(model_path, input_data, config, model_name)
                all_results.append(asdict(result))

                # 鐭殏鍐峰嵈
                time.sleep(2)

    # 淇濆瓨缁撴灉
    print(f"\n========== Ablation Experiment Summary ==========")
    print(f"{'model':<12} {'pool_size':<8} {'threads':<6} {'status':<10} {'throughput':<10} {'avg_latency':<10} "
          f"{'P99_latency':<10} {'peak_rss':<10} {'rss_drift':<10}")
    print("-" * 90)

    skipped_count = 0
    for r in all_results:
        cfg = r['config']
        if r.get('status') == 'SKIPPED':
            skipped_count += 1
            print(f"{r['model']:<12} {cfg['pool_size']:<8} {cfg['intra_threads']:<6} "
                  f"{'SKIPPED':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
        else:
            print(f"{r['model']:<12} {cfg['pool_size']:<8} {cfg['intra_threads']:<6} "
                  f"{r.get('status', 'OK'):<10} "
                  f"{r['throughput_reqs']:<10.2f} {r['avg_latency_ms']:<10.3f} "
                  f"{r['p99_latency_ms']:<10.3f} {r['peak_rss_mb']:<10.2f} "
                  f"{r['rss_drift_mb']:<10.2f}")

    # 淇濆瓨JSON缁撴灉
    result_file = os.path.join(results_dir, 'python_session_pool_ablation.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")
    ok_count = sum(1 for r in all_results if r.get('status', 'OK') == 'OK')
    print(f"Total {len(all_results)}  ablation experiments ({ok_count}  completed, {skipped_count}  skipped)")


if __name__ == '__main__':
    main()

