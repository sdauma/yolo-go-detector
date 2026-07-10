#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python_arena_ablation.py
CPU 内存 Arena 开关消融实验（Python 侧）

目的：与 Go 侧 go_arena_ablation.go 对应，验证 Unsafe Shared 架构的
      大幅 RSS 漂移是否来自共享 Session 的 CPU 内存分配器（Arena）。
      验证漂移机理跨语言一致性。

设计：
  - 控制变量：同模型(YOLO11x)、同并发度、同 intra_op/inter_op
  - 自变量：enable_mem_arena (True/False)，同时固定 enable_mem_pattern=False
  - 测试架构：Unsafe Shared (4并发) 与 Session Pool (池大小4)
  - 指标：吞吐量、峰值RSS、RSS漂移
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import psutil
import onnxruntime as ort


def get_process_rss():
    """获取进程级 RSS (MB)"""
    return psutil.Process(os.getpid()).memory_info().private / 1024 / 1024


def create_session(model_path, intra_op_threads=1, arena_enabled=True):
    """创建 ONNX Runtime Session，控制 arena 开关"""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 固定关闭 mem_pattern，避免额外预分配影响对比
    sess_options.enable_mem_pattern = False
    
    # 自变量：arena 开关
    # 注意：较新版本的 onnxruntime 支持 enable_mem_arena
    try:
        sess_options.enable_mem_arena = arena_enabled
    except AttributeError:
        # 如果版本不支持，通过日志告知
        print(f"  警告: 当前 onnxruntime 版本不支持 enable_mem_arena，arena={arena_enabled} 设置无效")
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    return session


def test_unsafe_shared(model_path, input_data, concurrency=4, num_requests=500,
                       intra_op_threads=1, arena_enabled=True):
    """测试 Unsafe Shared 架构（共享 Session，独立输入）"""
    arena_str = "ON" if arena_enabled else "OFF"
    print(f"测试 Unsafe Shared (arena={arena_str}): {concurrency} 并发，{num_requests} 请求")
    
    start_rss = get_process_rss()
    peak_rss = start_rss
    
    session = create_session(model_path, intra_op_threads, arena_enabled)
    input_name = session.get_inputs()[0].name
    
    latencies = []
    latency_lock = threading.Lock()
    peak_lock = threading.Lock()
    
    batch_size = num_requests // concurrency
    
    def worker():
        nonlocal peak_rss
        local_latencies = []
        for _ in range(batch_size):
            current_rss = get_process_rss()
            with peak_lock:
                if current_rss > peak_rss:
                    peak_rss = current_rss
            
            start = time.time()
            session.run(None, {input_name: input_data})
            latency = (time.time() - start) * 1000  # ms
            local_latencies.append(latency)
        
        with latency_lock:
            latencies.extend(local_latencies)
    
    threads = []
    start_time = time.time()
    
    for _ in range(concurrency):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = (time.time() - start_time) * 1000  # ms
    end_rss = get_process_rss()
    
    avg_latency = np.mean(latencies) if latencies else 0
    throughput = len(latencies) / (total_time / 1000) if total_time > 0 else 0
    
    return {
        'architecture': 'Unsafe Shared',
        'arena_enabled': arena_enabled,
        'concurrency': concurrency,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'start_rss': start_rss,
        'peak_rss': peak_rss,
        'end_rss': end_rss,
        'rss_drift': end_rss - start_rss,
    }


def test_session_pool(model_path, input_data, pool_size=4, num_requests=500,
                      intra_op_threads=1, arena_enabled=True):
    """测试 Session Pool 架构（独立 Session）"""
    arena_str = "ON" if arena_enabled else "OFF"
    print(f"测试 Session Pool (arena={arena_str}): pool_size={pool_size}, {num_requests} 请求")
    
    start_rss = get_process_rss()
    peak_rss = start_rss
    
    # 创建 Session 池
    session_pool = []
    input_names = []
    for _ in range(pool_size):
        sess = create_session(model_path, intra_op_threads, arena_enabled)
        session_pool.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    
    latencies = []
    latency_lock = threading.Lock()
    peak_lock = threading.Lock()
    
    batch_size = num_requests // pool_size
    
    def worker(sess, input_name):
        nonlocal peak_rss
        local_latencies = []
        for _ in range(batch_size):
            current_rss = get_process_rss()
            with peak_lock:
                if current_rss > peak_rss:
                    peak_rss = current_rss
            
            start = time.time()
            sess.run(None, {input_name: input_data})
            latency = (time.time() - start) * 1000
            local_latencies.append(latency)
        
        with latency_lock:
            latencies.extend(local_latencies)
    
    threads = []
    start_time = time.time()
    
    for i in range(pool_size):
        t = threading.Thread(target=worker, args=(session_pool[i], input_names[i]))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = (time.time() - start_time) * 1000
    end_rss = get_process_rss()
    
    avg_latency = np.mean(latencies) if latencies else 0
    throughput = len(latencies) / (total_time / 1000) if total_time > 0 else 0
    
    return {
        'architecture': 'Session Pool',
        'arena_enabled': arena_enabled,
        'concurrency': pool_size,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'start_rss': start_rss,
        'peak_rss': peak_rss,
        'end_rss': end_rss,
        'rss_drift': end_rss - start_rss,
    }


def main():
    model_path = "../../third_party/yolo11x.onnx"
    
    # 生成固定输入数据（与 Go 侧一致的形状）
    np.random.seed(12345)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    concurrency = 4
    pool_size = 4
    num_requests = 500
    intra_op_threads = 1
    
    print("===== Python CPU 内存 Arena 开关消融实验 =====")
    print(f"模型: YOLO11x, 并发度: {concurrency}, 池大小: {pool_size}, 请求总数: {num_requests}")
    print(f"intra_op={intra_op_threads}, inter_op=1, enable_mem_pattern=False(固定)")
    print()
    
    results = []
    
    # 实验 1: Unsafe Shared, arena=ON
    print("--- 实验 1: Unsafe Shared, arena=ON ---")
    r1 = test_unsafe_shared(model_path, input_data, concurrency, num_requests, intra_op_threads, True)
    results.append(r1)
    
    # 实验 2: Unsafe Shared, arena=OFF
    print("\n--- 实验 2: Unsafe Shared, arena=OFF ---")
    r2 = test_unsafe_shared(model_path, input_data, concurrency, num_requests, intra_op_threads, False)
    results.append(r2)
    
    # 实验 3: Session Pool, arena=ON
    print("\n--- 实验 3: Session Pool, arena=ON ---")
    r3 = test_session_pool(model_path, input_data, pool_size, num_requests, intra_op_threads, True)
    results.append(r3)
    
    # 实验 4: Session Pool, arena=OFF
    print("\n--- 实验 4: Session Pool, arena=OFF ---")
    r4 = test_session_pool(model_path, input_data, pool_size, num_requests, intra_op_threads, False)
    results.append(r4)
    
    # 输出结果
    print("\n===== Arena 开关消融实验结果 =====")
    print()
    print(f"{'架构':<16} {'Arena':<8} {'吞吐量':<12} {'平均延迟':<12} {'峰值RSS':<12} {'RSS漂移':<12}")
    print(f"{'':<16} {'':<8} {'(REQ/s)':<12} {'(ms)':<12} {'(MB)':<12} {'(MB)':<12}")
    print("-" * 72)
    
    for r in results:
        arena_str = "ON" if r['arena_enabled'] else "OFF"
        print(f"{r['architecture']:<16} {arena_str:<8} {r['throughput']:<12.5f} {r['avg_latency']:<12.3f} {r['peak_rss']:<12.2f} {r['rss_drift']:<12.2f}")
    
    # 保存到文件
    output_path = os.path.join("..", "..", "results", "python_arena_ablation_result.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== Python CPU 内存 Arena 开关消融实验结果 =====\n\n")
        f.write(f"模型: YOLO11x, 并发度: {concurrency}, 池大小: {pool_size}, 请求总数: {num_requests}\n")
        f.write(f"intra_op={intra_op_threads}, inter_op=1, enable_mem_pattern=False(固定)\n\n")
        
        for r in results:
            arena_str = "ON" if r['arena_enabled'] else "OFF"
            f.write(f"===== {r['architecture']} (arena={arena_str}) =====\n")
            f.write(f"  吞吐量: {r['throughput']:.5f} REQ/s\n")
            f.write(f"  平均延迟: {r['avg_latency']:.5f} ms\n")
            f.write(f"  起始RSS: {r['start_rss']:.5f} MB\n")
            f.write(f"  峰值RSS: {r['peak_rss']:.5f} MB\n")
            f.write(f"  结束RSS: {r['end_rss']:.5f} MB\n")
            f.write(f"  RSS漂移: {r['rss_drift']:.5f} MB\n\n")
    
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
