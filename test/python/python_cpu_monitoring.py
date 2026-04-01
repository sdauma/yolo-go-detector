# python_cpu_monitoring.py
# Python CPU 监控测试 - 监测推理过程中的 CPU 使用率
#
# 测试目的：
# - 监测 ONNX Runtime 推理时的 CPU 使用率
# - 分析不同并发配置下的 CPU 负载分布
# - 为论文提供 CPU 利用率数据

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass
from typing import List
import threading

# 固定随机种子，确保可复现
np.random.seed(12345)

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型路径
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

print("===== Python CPU 监控测试 =====")
print(f"模型路径: {model_path}")

# 创建输入数据
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 获取 CPU 信息
print(f"\n系统 CPU 信息:")
print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
print(f"  逻辑核心数: {psutil.cpu_count(logical=True)}")
print(f"  当前 CPU 频率: {psutil.cpu_freq().current:.0f} MHz")

# 创建 Session
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

print(f"\n创建 InferenceSession...")
session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 预热
print("预热...")
for _ in range(5):
    session.run(None, {input_name: input_data})

# 监测 CPU 使用率
print("\n开始监测 CPU 使用率...")
num_requests = 50
cpu_samples = []

# 获取初始 CPU 使用率
process = psutil.Process()
initial_cpu_percent = process.cpu_percent()
initial_time = time.time()

for i in range(num_requests):
    # 记录推理前的 CPU 使用率
    cpu_before = process.cpu_percent()
    
    # 执行推理
    start = time.time()
    session.run(None, {input_name: input_data})
    latency = (time.time() - start) * 1000  # ms
    
    # 记录推理后的 CPU 使用率
    cpu_after = process.cpu_percent()
    
    cpu_samples.append({
        'request': i + 1,
        'latency': latency,
        'cpu_before': cpu_before,
        'cpu_after': cpu_after
    })
    
    if (i + 1) % 10 == 0:
        print(f"  完成 {i+1}/{num_requests} 次推理")

# 计算统计信息
latencies = [s['latency'] for s in cpu_samples]
cpu_afters = [s['cpu_after'] for s in cpu_samples]

avg_latency = sum(latencies) / len(latencies)
avg_cpu = sum(cpu_afters) / len(cpu_afters)
max_cpu = max(cpu_afters)
min_cpu = min(cpu_afters)

print(f"\n===== 测试结果 =====")
print(f"总请求数: {num_requests}")
print(f"平均延迟: {avg_latency:.2f} ms")
print(f"平均 CPU 使用率: {avg_cpu:.2f}%")
print(f"峰值 CPU 使用率: {max_cpu:.2f}%")
print(f"最低 CPU 使用率: {min_cpu:.2f}%")

# 保存结果
result_path = os.path.join(base_path, "results", "python_cpu_monitoring_result.txt")
os.makedirs(os.path.dirname(result_path), exist_ok=True)

with open(result_path, 'w', encoding='utf-8') as f:
    f.write("===== Python CPU 监控测试结果 =====\n\n")
    f.write(f"模型: YOLO11x\n")
    f.write(f"输入尺寸: 1x3x640x640\n")
    f.write(f"intra_op_num_threads: 1\n")
    f.write(f"inter_op_num_threads: 1\n\n")
    f.write(f"系统信息:\n")
    f.write(f"  物理核心数: {psutil.cpu_count(logical=False)}\n")
    f.write(f"  逻辑核心数: {psutil.cpu_count(logical=True)}\n")
    f.write(f"  当前 CPU 频率: {psutil.cpu_freq().current:.0f} MHz\n\n")
    f.write(f"性能指标:\n")
    f.write(f"  总请求数: {num_requests}\n")
    f.write(f"  平均延迟: {avg_latency:.2f} ms\n")
    f.write(f"  平均 CPU 使用率: {avg_cpu:.2f}%\n")
    f.write(f"  峰值 CPU 使用率: {max_cpu:.2f}%\n")
    f.write(f"  最低 CPU 使用率: {min_cpu:.2f}%\n\n")
    f.write("详细数据:\n")
    f.write("请求号, 延迟(ms), CPU使用率(%)\n")
    for s in cpu_samples:
        f.write(f"{s['request']}, {s['latency']:.2f}, {s['cpu_after']:.2f}\n")

print(f"\n结果已保存到: {result_path}")
print("===== 测试完成 =====")
