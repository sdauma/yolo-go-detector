# python_cold_start_decomposition.py
# Python 冷启动分解测试
# 
# 测试目的：
# - 分解冷启动时间为会话创建时间、模型加载时间和首次推理时间
# - 执行20次冷启动测试，计算平均值
# - 确保数据稳定性和可重复性

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# 固定随机种子，确保可复现
np.random.seed(12345)

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型路径 - 同时测试大模型和轻模型
model_path_large = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
model_path_small = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 检查模型文件是否存在
if not os.path.exists(model_path_large):
    print(f"错误: 大模型文件不存在: {model_path_large}")
    sys.exit(1)
if not os.path.exists(model_path_small):
    print(f"错误: 轻模型文件不存在: {model_path_small}")
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
    print(f"\n===== Python 冷启动测试 - {model_name} ====")
    
    # 绑定CPU核心
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1, 2, 3])
    
    # 记录初始内存
    start_rss = process.memory_info().rss / 1024 / 1024
    
    # 1. 会话创建时间
    print("创建 InferenceSession...")
    t0 = time.perf_counter()
    
    try:
        sess_options = ort.SessionOptions()
        
        # 显式设置所有 SessionOptions 参数
        # 线程配置 - 12线程，与其他测试保持一致
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
        print(f"错误: 创建 InferenceSession 失败: {e}")
        sys.exit(1)
    
    t1 = time.perf_counter()
    session_creation_time = (t1 - t0) * 1000
    
    # 2. 模型加载时间（会话创建已经包含了模型加载）
    # 注：在ONNX Runtime中，模型加载是在会话创建时完成的
    model_loading_time = session_creation_time
    
    # 获取输入信息
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # 使用与 Go 完全一致的输入数据（从文件加载，使用固定种子）
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
    except Exception as e:
        print(f"加载输入数据失败: {e}")
        sys.exit(1)

    # 3. 首次推理时间
    print("执行首次推理...")
    t2 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t3 = time.perf_counter()
    first_inference_time = (t3 - t2) * 1000
    
    # 计算总冷启动时间
    total_cold_start_time = session_creation_time + first_inference_time
    
    # 记录峰值内存
    peak_rss = process.memory_info().rss / 1024 / 1024
    
    print(f"会话创建时间: {session_creation_time:.3f} ms")
    print(f"模型加载时间: {model_loading_time:.3f} ms")
    print(f"首次推理时间: {first_inference_time:.3f} ms")
    print(f"总冷启动时间: {total_cold_start_time:.3f} ms")
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
    print("===== Python 冷启动分解测试（20次运行）=====")

    # 运行20次测试 - 大模型
    num_runs = 20
    results_large = []
    results_small = []

    print("\n===== 测试大模型 (YOLO11x) =====")
    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 次测试 =====")
        result = run_cold_start_test(model_path_large, "YOLO11x")
        results_large.append(result)

    print("\n===== 测试轻模型 (YOLO11n) =====")
    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 次测试 =====")
        result = run_cold_start_test(model_path_small, "YOLO11n")
        results_small.append(result)

    # 计算大模型平均值
    avg_session_creation_large = sum(r.session_creation_time for r in results_large) / num_runs
    avg_model_loading_large = sum(r.model_loading_time for r in results_large) / num_runs
    avg_first_inference_large = sum(r.first_inference_time for r in results_large) / num_runs
    avg_total_cold_start_large = sum(r.total_cold_start_time for r in results_large) / num_runs
    avg_start_rss_large = sum(r.start_rss for r in results_large) / num_runs
    avg_peak_rss_large = sum(r.peak_rss for r in results_large) / num_runs

    # 计算轻模型平均值
    avg_session_creation_small = sum(r.session_creation_time for r in results_small) / num_runs
    avg_model_loading_small = sum(r.model_loading_time for r in results_small) / num_runs
    avg_first_inference_small = sum(r.first_inference_time for r in results_small) / num_runs
    avg_total_cold_start_small = sum(r.total_cold_start_time for r in results_small) / num_runs
    avg_start_rss_small = sum(r.start_rss for r in results_small) / num_runs
    avg_peak_rss_small = sum(r.peak_rss for r in results_small) / num_runs

    print("\n===== 大模型 (YOLO11x) 20次测试平均值 =====")
    print(f"会话创建时间: {avg_session_creation_large:.3f} ms")
    print(f"模型加载时间: {avg_model_loading_large:.3f} ms")
    print(f"首次推理时间: {avg_first_inference_large:.3f} ms")
    print(f"总冷启动时间: {avg_total_cold_start_large:.3f} ms")
    print(f"Start RSS: {avg_start_rss_large:.2f} MB")
    print(f"Peak RSS: {avg_peak_rss_large:.2f} MB")

    print("\n===== 轻模型 (YOLO11n) 20次测试平均值 =====")
    print(f"会话创建时间: {avg_session_creation_small:.3f} ms")
    print(f"模型加载时间: {avg_model_loading_small:.3f} ms")
    print(f"首次推理时间: {avg_first_inference_small:.3f} ms")
    print(f"总冷启动时间: {avg_total_cold_start_small:.3f} ms")
    print(f"Start RSS: {avg_start_rss_small:.2f} MB")
    print(f"Peak RSS: {avg_peak_rss_small:.2f} MB")

    # 保存结果
    result_path = os.path.join(base_path, "results", "python_cold_start_decomposition_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python 冷启动分解测试结果 =====\n\n")
        
        f.write("===== 大模型 (YOLO11x) =====\n")
        for i, r in enumerate(results_large):
            f.write(f"===== 第 {i+1} 次测试 =====\n")
            f.write(f"会话创建时间: {r.session_creation_time:.5f} ms\n")
            f.write(f"模型加载时间: {r.model_loading_time:.5f} ms\n")
            f.write(f"首次推理时间: {r.first_inference_time:.5f} ms\n")
            f.write(f"总冷启动时间: {r.total_cold_start_time:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n\n")
        
        f.write("===== 大模型 (YOLO11x) 20次测试平均值 =====\n")
        f.write(f"会话创建时间: {avg_session_creation_large:.5f} ms\n")
        f.write(f"模型加载时间: {avg_model_loading_large:.5f} ms\n")
        f.write(f"首次推理时间: {avg_first_inference_large:.5f} ms\n")
        f.write(f"总冷启动时间: {avg_total_cold_start_large:.5f} ms\n")
        f.write(f"Start RSS: {avg_start_rss_large:.5f} MB\n")
        f.write(f"Peak RSS: {avg_peak_rss_large:.5f} MB\n\n")
        
        f.write("===== 轻模型 (YOLO11n) =====\n")
        for i, r in enumerate(results_small):
            f.write(f"===== 第 {i+1} 次测试 =====\n")
            f.write(f"会话创建时间: {r.session_creation_time:.5f} ms\n")
            f.write(f"模型加载时间: {r.model_loading_time:.5f} ms\n")
            f.write(f"首次推理时间: {r.first_inference_time:.5f} ms\n")
            f.write(f"总冷启动时间: {r.total_cold_start_time:.5f} ms\n")
            f.write(f"Start RSS: {r.start_rss:.5f} MB\n")
            f.write(f"Peak RSS: {r.peak_rss:.5f} MB\n\n")
        
        f.write("===== 轻模型 (YOLO11n) 20次测试平均值 =====\n")
        f.write(f"会话创建时间: {avg_session_creation_small:.5f} ms\n")
        f.write(f"模型加载时间: {avg_model_loading_small:.5f} ms\n")
        f.write(f"首次推理时间: {avg_first_inference_small:.5f} ms\n")
        f.write(f"总冷启动时间: {avg_total_cold_start_small:.5f} ms\n")
        f.write(f"Start RSS: {avg_start_rss_small:.5f} MB\n")
        f.write(f"Peak RSS: {avg_peak_rss_small:.5f} MB\n")

    print(f"\n结果已保存到: {result_path}")
    print("测试完成!")

if __name__ == "__main__":
    main()
