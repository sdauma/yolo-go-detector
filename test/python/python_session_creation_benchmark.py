# python_session_creation_benchmark.py
# Python Session创建时间测试
#
# 测试目的：
# - 测量Python创建InferenceSession的时间
# - 与Go的Session创建时间进行对比
# - 提供客观的跨语言对比数据

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

# 构建模型路径
yolo11x_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
yolo11n_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

@dataclass
class SessionCreationResult:
    avg_time: float
    std_time: float
    p50_time: float
    p90_time: float
    min_time: float
    max_time: float
    times: list

def run_session_creation_benchmark(model_name, model_path):
    print(f"===== Python Session创建时间测试 - {model_name} ====")
    
    # 不绑定CPU核心，让系统自由调度（匹配Go的默认行为）
    process = psutil.Process(os.getpid())
    print("CPU核心调度：系统默认")
    
    # 测试Session创建时间
    print(f"测试{model_name}模型的Session创建时间...")
    runs = 100  # 创建100次Session
    times = []

    for i in range(runs):
        t0 = time.perf_counter()
        try:
            sess_options = ort.SessionOptions()
            # 线程配置 - 12线程，匹配Go的默认行为
            sess_options.intra_op_num_threads = 12
            sess_options.inter_op_num_threads = 1
            # 日志配置（关闭所有日志）
            sess_options.log_severity_level = 3
            # 性能分析配置（关闭性能分析）
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # 内存池配置（启用内存池复用）
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
        dt = (t1 - t0) * 1000  # 转换为毫秒
        times.append(dt)
        
        # 释放Session资源
        del sess

    # 计算结果
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)
    p50_time = np.percentile(times, 50)
    p90_time = np.percentile(times, 90)

    return SessionCreationResult(
        avg_time=avg_time,
        std_time=std_time,
        p50_time=p50_time,
        p90_time=p90_time,
        min_time=min_time,
        max_time=max_time,
        times=times
    )

def main():
    print("===== Python Session创建时间测试 ====")
    print("测试配置：")
    print("- 线程数: 12")
    print("- 创建次数: 100次")
    print()

    # 测试 YOLO11x 模型
    print("\n===== 测试 YOLO11x 模型 =====")
    if not os.path.exists(yolo11x_path):
        print(f"错误: YOLO11x模型文件不存在: {yolo11x_path}")
        sys.exit(1)
    
    yolo11x_result = run_session_creation_benchmark("YOLO11x", yolo11x_path)
    
    print(f"\nYOLO11x Session创建时间结果:")
    print(f"平均时间: {yolo11x_result.avg_time:.3f} ms")
    print(f"标准差: {yolo11x_result.std_time:.3f} ms")
    print(f"P50时间: {yolo11x_result.p50_time:.3f} ms")
    print(f"P90时间: {yolo11x_result.p90_time:.3f} ms")
    print(f"最小时间: {yolo11x_result.min_time:.3f} ms")
    print(f"最大时间: {yolo11x_result.max_time:.3f} ms")

    # 测试 YOLO11n 模型
    print("\n===== 测试 YOLO11n 模型 =====")
    if not os.path.exists(yolo11n_path):
        print(f"错误: YOLO11n模型文件不存在: {yolo11n_path}")
        sys.exit(1)
    
    yolo11n_result = run_session_creation_benchmark("YOLO11n", yolo11n_path)
    
    print(f"\nYOLO11n Session创建时间结果:")
    print(f"平均时间: {yolo11n_result.avg_time:.3f} ms")
    print(f"标准差: {yolo11n_result.std_time:.3f} ms")
    print(f"P50时间: {yolo11n_result.p50_time:.3f} ms")
    print(f"P90时间: {yolo11n_result.p90_time:.3f} ms")
    print(f"最小时间: {yolo11n_result.min_time:.3f} ms")
    print(f"最大时间: {yolo11n_result.max_time:.3f} ms")

    # 保存结果
    result_path = os.path.join(base_path, "results", "python_session_creation_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Session创建时间测试结果 =====\n")
        f.write("测试配置：\n")
        f.write("- 线程数: 12\n")
        f.write("- 创建次数: 100次\n")
        f.write("\n")
        
        f.write("===== YOLO11x 测试结果 =====\n")
        f.write(f"平均时间: {yolo11x_result.avg_time:.5f} ms\n")
        f.write(f"标准差: {yolo11x_result.std_time:.5f} ms\n")
        f.write(f"P50时间: {yolo11x_result.p50_time:.5f} ms\n")
        f.write(f"P90时间: {yolo11x_result.p90_time:.5f} ms\n")
        f.write(f"最小时间: {yolo11x_result.min_time:.5f} ms\n")
        f.write(f"最大时间: {yolo11x_result.max_time:.5f} ms\n")
        f.write("\n")
        
        f.write("===== YOLO11n 测试结果 =====\n")
        f.write(f"平均时间: {yolo11n_result.avg_time:.5f} ms\n")
        f.write(f"标准差: {yolo11n_result.std_time:.5f} ms\n")
        f.write(f"P50时间: {yolo11n_result.p50_time:.5f} ms\n")
        f.write(f"P90时间: {yolo11n_result.p90_time:.5f} ms\n")
        f.write(f"最小时间: {yolo11n_result.min_time:.5f} ms\n")
        f.write(f"最大时间: {yolo11n_result.max_time:.5f} ms\n")

    print(f"\n结果已保存到: {result_path}")
    print("测试完成!")

if __name__ == "__main__":
    main()
