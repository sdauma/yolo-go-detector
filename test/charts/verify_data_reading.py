# 验证数据读取是否正确
import os
import json

# 正确的 results 目录路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")

def read_architecture_result(file_path):
    """读取架构对比结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取 12 并发的吞吐量数据
        unsafe_throughput = None
        mutex_throughput = None
        pool_throughput = None
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '===== Unsafe Shared =====' in line and i + 5 < len(lines):
                if '并发度: 12' in lines[i+1]:
                    for j in range(i+2, i+10):
                        if '吞吐量:' in lines[j]:
                            parts = lines[j].split('吞吐量:')
                            if len(parts) == 2:
                                unsafe_throughput = float(parts[1].strip().split(' ')[0])
                
            elif '===== Mutex Shared =====' in line and i + 5 < len(lines):
                if '并发度: 12' in lines[i+1]:
                    for j in range(i+2, i+10):
                        if '吞吐量:' in lines[j]:
                            parts = lines[j].split('吞吐量:')
                            if len(parts) == 2:
                                mutex_throughput = float(parts[1].strip().split(' ')[0])
                
            elif '===== Session Pool =====' in line and i + 5 < len(lines):
                if '池大小: 12' in lines[i+1]:
                    for j in range(i+2, i+10):
                        if '吞吐量:' in lines[j]:
                            parts = lines[j].split('吞吐量:')
                            if len(parts) == 2:
                                pool_throughput = float(parts[1].strip().split(' ')[0])
        
        return [unsafe_throughput, mutex_throughput, pool_throughput]
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    return None

def read_batch_result(file_path):
    """读取 batch 测试结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        batch_sizes = []
        throughput = []
        latency = []
        
        for item in data.get('results', []):
            batch_sizes.append(item.get('batch_size'))
            throughput.append(item.get('throughput_images_per_sec'))
            latency.append(item.get('per_image_time_ms'))
        
        return batch_sizes, throughput, latency
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    return None, None, None

def read_stability_result(file_path):
    """读取稳定性测试结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        rss_values = []
        for line in lines[1:]:  # 跳过表头
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        rss = float(parts[2])
                        rss_values.append(rss)
                    except:
                        continue
        
        # 生成时间点（每5分钟一个点）
        time_points = np.arange(0, len(rss_values) * 5, 5)
        if len(time_points) > 7:  # 限制最多7个点
            step = len(time_points) // 7
            time_points = time_points[::step][:7]
            rss_values = rss_values[::step][:7]
        
        return time_points, rss_values
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    return None, None

import numpy as np

print("=" * 60)
print("验证数据读取结果")
print("=" * 60)

# 验证图1数据
print("\n【图1】架构对比数据：")
arch_data = read_architecture_result(os.path.join(results_dir, "go_architecture_comparison.txt"))
print(f"  Unsafe Shared 12并发: {arch_data[0]}")
print(f"  Mutex Shared 12并发: {arch_data[1]}")
print(f"  Session Pool 12并发: {arch_data[2]}")
print(f"  读取结果: {arch_data}")

# 验证图3数据
print("\n【图3】Batch测试数据：")
batch_sizes, batch_throughput, batch_latency = read_batch_result(os.path.join(results_dir, "go_batch_inference_result.json"))
print(f"  Batch Sizes: {batch_sizes}")
print(f"  延迟 (ms): {batch_latency}")
print(f"  吞吐量 (img/s): {batch_throughput}")

# 验证图6数据
print("\n【图6】稳定性测试数据：")
stability_time, stability_rss = read_stability_result(os.path.join(results_dir, "go_long_stability_detailed.csv"))
print(f"  时间点: {stability_time}")
print(f"  RSS值: {stability_rss}")
print(f"  数据点数量: {len(stability_time)}")

print("\n" + "=" * 60)
