# python_memory_standardization.py
# Python 内存标准化测试
# 
# 测试目的：
# - 记录解释器常驻内存（基础内存）
# - 记录模型加载后的内存
# - 记录推理后的内存
# - 执行多次测试，计算平均值
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
class MemoryResult:
    interpreter_memory: float  # 解释器常驻内存
    model_loaded_memory: float  # 模型加载后内存
    post_inference_memory: float  # 推理后内存
    memory_increase: float  # 内存增加量（模型加载+推理）

def run_memory_test(model_path, model_name):
    print(f"\n===== Python 内存测试 - {model_name} ====")
    
    # 绑定CPU核心
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1, 2, 3])
    
    # 1. 测量解释器常驻内存（基础内存）
    print("测量解释器常驻内存...")
    interpreter_memory = process.memory_info().rss / 1024 / 1024
    print(f"解释器常驻内存: {interpreter_memory:.2f} MB")
    
    # 2. 创建 Session（加载模型）
    print("加载模型...")
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
    
    # 3. 测量模型加载后的内存
    model_loaded_memory = process.memory_info().rss / 1024 / 1024
    print(f"模型加载后内存: {model_loaded_memory:.2f} MB")
    
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

    # 4. 执行推理
    print("执行推理...")
    for _ in range(10):  # 执行10次推理以稳定内存使用
        sess.run(None, {input_name: input_data})
    
    # 5. 测量推理后的内存
    post_inference_memory = process.memory_info().rss / 1024 / 1024
    print(f"推理后内存: {post_inference_memory:.2f} MB")
    
    # 计算内存增加量
    memory_increase = post_inference_memory - interpreter_memory
    print(f"内存增加量: {memory_increase:.2f} MB")
    
    return MemoryResult(
        interpreter_memory=interpreter_memory,
        model_loaded_memory=model_loaded_memory,
        post_inference_memory=post_inference_memory,
        memory_increase=memory_increase
    )

def main():
    print("===== Python 内存标准化测试（10次运行）=====")

    # 运行10次测试 - 大模型
    num_runs = 10
    results_large = []
    results_small = []

    print("\n===== 测试大模型 (YOLO11x) =====")
    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 次测试 =====")
        result = run_memory_test(model_path_large, "YOLO11x")
        results_large.append(result)

    print("\n===== 测试轻模型 (YOLO11n) =====")
    for i in range(num_runs):
        print(f"\n===== 第 {i+1} 次测试 =====")
        result = run_memory_test(model_path_small, "YOLO11n")
        results_small.append(result)

    # 计算大模型平均值
    avg_interpreter_large = sum(r.interpreter_memory for r in results_large) / num_runs
    avg_model_loaded_large = sum(r.model_loaded_memory for r in results_large) / num_runs
    avg_post_inference_large = sum(r.post_inference_memory for r in results_large) / num_runs
    avg_memory_increase_large = sum(r.memory_increase for r in results_large) / num_runs

    # 计算轻模型平均值
    avg_interpreter_small = sum(r.interpreter_memory for r in results_small) / num_runs
    avg_model_loaded_small = sum(r.model_loaded_memory for r in results_small) / num_runs
    avg_post_inference_small = sum(r.post_inference_memory for r in results_small) / num_runs
    avg_memory_increase_small = sum(r.memory_increase for r in results_small) / num_runs

    print("\n===== 大模型 (YOLO11x) 10次测试平均值 =====")
    print(f"解释器常驻内存: {avg_interpreter_large:.2f} MB")
    print(f"模型加载后内存: {avg_model_loaded_large:.2f} MB")
    print(f"推理后内存: {avg_post_inference_large:.2f} MB")
    print(f"内存增加量: {avg_memory_increase_large:.2f} MB")

    print("\n===== 轻模型 (YOLO11n) 10次测试平均值 =====")
    print(f"解释器常驻内存: {avg_interpreter_small:.2f} MB")
    print(f"模型加载后内存: {avg_model_loaded_small:.2f} MB")
    print(f"推理后内存: {avg_post_inference_small:.2f} MB")
    print(f"内存增加量: {avg_memory_increase_small:.2f} MB")

    # 保存结果
    result_path = os.path.join(base_path, "results", "python_memory_standardization_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python 内存标准化测试结果 =====\n\n")
        
        f.write("===== 大模型 (YOLO11x) =====\n")
        for i, r in enumerate(results_large):
            f.write(f"===== 第 {i+1} 次测试 =====\n")
            f.write(f"解释器常驻内存: {r.interpreter_memory:.5f} MB\n")
            f.write(f"模型加载后内存: {r.model_loaded_memory:.5f} MB\n")
            f.write(f"推理后内存: {r.post_inference_memory:.5f} MB\n")
            f.write(f"内存增加量: {r.memory_increase:.5f} MB\n\n")
        
        f.write("===== 大模型 (YOLO11x) 10次测试平均值 =====\n")
        f.write(f"解释器常驻内存: {avg_interpreter_large:.5f} MB\n")
        f.write(f"模型加载后内存: {avg_model_loaded_large:.5f} MB\n")
        f.write(f"推理后内存: {avg_post_inference_large:.5f} MB\n")
        f.write(f"内存增加量: {avg_memory_increase_large:.5f} MB\n\n")
        
        f.write("===== 轻模型 (YOLO11n) =====\n")
        for i, r in enumerate(results_small):
            f.write(f"===== 第 {i+1} 次测试 =====\n")
            f.write(f"解释器常驻内存: {r.interpreter_memory:.5f} MB\n")
            f.write(f"模型加载后内存: {r.model_loaded_memory:.5f} MB\n")
            f.write(f"推理后内存: {r.post_inference_memory:.5f} MB\n")
            f.write(f"内存增加量: {r.memory_increase:.5f} MB\n\n")
        
        f.write("===== 轻模型 (YOLO11n) 10次测试平均值 =====\n")
        f.write(f"解释器常驻内存: {avg_interpreter_small:.5f} MB\n")
        f.write(f"模型加载后内存: {avg_model_loaded_small:.5f} MB\n")
        f.write(f"推理后内存: {avg_post_inference_small:.5f} MB\n")
        f.write(f"内存增加量: {avg_memory_increase_small:.5f} MB\n")

    print(f"\n结果已保存到: {result_path}")
    print("测试完成!")

if __name__ == "__main__":
    main()
