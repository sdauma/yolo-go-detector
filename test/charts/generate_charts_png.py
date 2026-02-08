#!/usr/bin/env python3
# generate_charts_png.py
# 生成所有测试结果的PNG格式图表

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results")
charts_dir = os.path.join(results_dir, "charts")

# 确保charts目录存在
os.makedirs(charts_dir, exist_ok=True)

def read_cold_start_results():
    """读取冷启动测试结果"""
    go_cold_start_path = os.path.join(results_dir, "go_cold_start_result.txt")
    py_cold_start_path = os.path.join(results_dir, "python_cold_start_result.txt")
    
    go_cold_start_time = 0
    go_stable_time = 0
    py_cold_start_time = 0
    py_stable_time = 0
    
    # 解析Go冷启动测试结果
    if os.path.exists(go_cold_start_path):
        with open(go_cold_start_path, "r", encoding="utf-8") as f:
            content = f.read()
            cold_start_match = re.search(r"冷启动时间: ([0-9.]+) ms", content)
            if cold_start_match:
                go_cold_start_time = float(cold_start_match.group(1))
            stable_match = re.search(r"稳定状态时间: ([0-9.]+) ms", content)
            if stable_match:
                go_stable_time = float(stable_match.group(1))
    
    # 解析Python冷启动测试结果
    if os.path.exists(py_cold_start_path):
        with open(py_cold_start_path, "r", encoding="utf-8") as f:
            content = f.read()
            cold_start_match = re.search(r"冷启动时间: ([0-9.]+) ms", content)
            if cold_start_match:
                py_cold_start_time = float(cold_start_match.group(1))
            stable_match = re.search(r"稳定状态时间: ([0-9.]+) ms", content)
            if stable_match:
                py_stable_time = float(stable_match.group(1))
    
    return {
        "go": {
            "cold_start": go_cold_start_time,
            "stable": go_stable_time
        },
        "python": {
            "cold_start": py_cold_start_time,
            "stable": py_stable_time
        }
    }

def read_thread_config_results():
    """读取线程配置测试结果"""
    threads = [1, 2, 4, 8]
    go_results = {}
    py_results = {}
    
    for thread in threads:
        # 读取Go线程配置测试结果
        go_thread_path = os.path.join(results_dir, f"go_thread_{thread}_result.txt")
        if os.path.exists(go_thread_path):
            with open(go_thread_path, "r", encoding="utf-8") as f:
                content = f.read()
                avg_match = re.search(r"平均延迟: ([0-9.]+) ms", content)
                if avg_match:
                    go_results[thread] = float(avg_match.group(1))
        
        # 读取Python线程配置测试结果
        py_thread_path = os.path.join(results_dir, f"python_thread_{thread}_result.txt")
        if os.path.exists(py_thread_path):
            with open(py_thread_path, "r", encoding="utf-8") as f:
                content = f.read()
                avg_match = re.search(r"平均延迟: ([0-9.]+) ms", content)
                if avg_match:
                    py_results[thread] = float(avg_match.group(1))
    
    return {
        "go": go_results,
        "python": py_results
    }

def read_thread_config_comprehensive():
    """读取线程配置综合结果"""
    go_comprehensive_path = os.path.join(results_dir, "go_thread_config_comprehensive.txt")
    py_comprehensive_path = os.path.join(results_dir, "python_thread_config_comprehensive.txt")
    
    go_data = {}
    py_data = {}
    
    # 解析Go综合结果
    if os.path.exists(go_comprehensive_path):
        with open(go_comprehensive_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[2:]:  # 跳过前两行标题
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        thread = int(parts[0])
                        go_data[thread] = {
                            'avg_latency': float(parts[1]),
                            'std_dev': float(parts[2]),
                            'fps': float(parts[4]),
                            'p50': float(parts[5]),
                            'p90': float(parts[6]),
                            'p99': float(parts[7]),
                            'start_rss': float(parts[8]),
                            'stable_rss': float(parts[9])
                        }
                    except (ValueError, IndexError):
                        continue
    
    # 解析Python综合结果
    if os.path.exists(py_comprehensive_path):
        with open(py_comprehensive_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[2:]:  # 跳过前两行标题
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        thread = int(parts[0])
                        py_data[thread] = {
                            'avg_latency': float(parts[1]),
                            'std_dev': float(parts[2]),
                            'fps': float(parts[4]),
                            'p50': float(parts[5]),
                            'p90': float(parts[6]),
                            'p99': float(parts[7]),
                            'start_rss': float(parts[8]),
                            'stable_rss': float(parts[9])
                        }
                    except (ValueError, IndexError):
                        continue
    
    return {"go": go_data, "python": py_data}

def generate_cold_start_factor_chart(cold_start_results):
    """生成冷启动因子分析图表"""
    labels = ['Go', 'Python']
    go_factor = cold_start_results['go']['cold_start'] / cold_start_results['go']['stable'] if cold_start_results['go']['stable'] > 0 else 0
    py_factor = cold_start_results['python']['cold_start'] / cold_start_results['python']['stable'] if cold_start_results['python']['stable'] > 0 else 0
    
    factors = [go_factor, py_factor]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, factors, color=['#66c2a5', '#fc8d62'], alpha=0.8)
    
    ax.set_ylabel('Cold Start Factor (Cold Start / Stable)', fontsize=12)
    ax.set_title('Cold Start Factor Analysis', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(factors) * 1.2)
    
    # 在柱状图上添加数值标签
    for bar, factor in zip(bars, factors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{factor:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "cold_start_factor.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"冷启动因子分析图表已保存到: {output_path}")
    plt.close()

def generate_cold_start_vs_stable_chart(cold_start_results):
    """生成冷启动与稳定状态时间对比图表"""
    labels = ['Go', 'Python']
    cold_start_times = [cold_start_results['go']['cold_start'], cold_start_results['python']['cold_start']]
    stable_times = [cold_start_results['go']['stable'], cold_start_results['python']['stable']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, cold_start_times, width, label='Cold Start Time', color='#66c2a5', alpha=0.8)
    rects2 = ax.bar(x + width/2, stable_times, width, label='Stable Time', color='#fc8d62', alpha=0.8)
    
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Cold Start vs Stable Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "cold_start_vs_stable.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"冷启动与稳定状态对比图表已保存到: {output_path}")
    plt.close()

def generate_thread_config_avg_latency_chart(thread_config_results):
    """生成线程配置平均延迟图表"""
    threads = list(thread_config_results['go'].keys())
    go_times = list(thread_config_results['go'].values())
    py_times = list(thread_config_results['python'].values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(threads, go_times, marker='o', label='Go', linewidth=2, markersize=8, color='#66c2a5')
    ax.plot(threads, py_times, marker='s', label='Python', linewidth=2, markersize=8, color='#fc8d62')
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Average Latency vs Number of Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(fontsize=11)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "thread_config_avg_latency.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"线程配置平均延迟图表已保存到: {output_path}")
    plt.close()

def generate_thread_config_speedup_chart(comprehensive_data):
    """生成线程配置加速比图表"""
    go_data = comprehensive_data['go']
    py_data = comprehensive_data['python']
    
    if not go_data:
        print("警告: Go线程配置数据为空，跳过加速比图表生成")
        return
    
    threads = sorted(go_data.keys())
    go_baseline = go_data[1]['avg_latency']
    
    go_speedups = [go_baseline / go_data[t]['avg_latency'] for t in threads]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(threads, go_speedups, marker='o', label='Go', linewidth=2, markersize=8, color='#66c2a5')
    
    if py_data:
        py_baseline = py_data[1]['avg_latency']
        py_speedups = [py_baseline / py_data[t]['avg_latency'] for t in threads]
        ax.plot(threads, py_speedups, marker='s', label='Python', linewidth=2, markersize=8, color='#fc8d62')
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup (vs 1 thread)', fontsize=12)
    ax.set_title('Speedup vs Number of Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(fontsize=11)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "thread_config_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"线程配置加速比图表已保存到: {output_path}")
    plt.close()

def generate_thread_config_memory_usage_chart(comprehensive_data):
    """生成线程配置内存使用图表"""
    go_data = comprehensive_data['go']
    py_data = comprehensive_data['python']
    
    if not go_data:
        print("警告: Go线程配置数据为空，跳过内存使用图表生成")
        return
    
    threads = sorted(go_data.keys())
    go_rss = [go_data[t]['stable_rss'] for t in threads]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(threads, go_rss, marker='o', label='Go', linewidth=2, markersize=8, color='#66c2a5')
    
    if py_data:
        py_rss = [py_data[t]['stable_rss'] for t in threads]
        ax.plot(threads, py_rss, marker='s', label='Python', linewidth=2, markersize=8, color='#fc8d62')
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Stable RSS (MB)', fontsize=12)
    ax.set_title('Memory Usage vs Number of Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(fontsize=11)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "thread_config_memory_usage.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"线程配置内存使用图表已保存到: {output_path}")
    plt.close()

def generate_thread_config_latency_distribution_chart(comprehensive_data):
    """生成线程配置延迟分布图表"""
    go_data = comprehensive_data['go']
    py_data = comprehensive_data['python']
    
    if not go_data:
        print("警告: Go线程配置数据为空，跳过延迟分布图表生成")
        return
    
    threads = sorted(go_data.keys())
    go_p50 = [go_data[t]['p50'] for t in threads]
    go_p90 = [go_data[t]['p90'] for t in threads]
    go_p99 = [go_data[t]['p99'] for t in threads]
    
    if py_data:
        py_p50 = [py_data[t]['p50'] for t in threads]
        py_p90 = [py_data[t]['p90'] for t in threads]
        py_p99 = [py_data[t]['p99'] for t in threads]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    
    # Go延迟分布
    ax1.plot(threads, go_p50, marker='o', label='P50', linewidth=2, markersize=8)
    ax1.plot(threads, go_p90, marker='s', label='P90', linewidth=2, markersize=8)
    ax1.plot(threads, go_p99, marker='^', label='P99', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Go: Latency Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(threads)
    ax1.legend(fontsize=11)
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Python延迟分布
    if py_data:
        ax2.plot(threads, py_p50, marker='o', label='P50', linewidth=2, markersize=8)
        ax2.plot(threads, py_p90, marker='s', label='P90', linewidth=2, markersize=8)
        ax2.plot(threads, py_p99, marker='^', label='P99', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Threads', fontsize=12)
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Python: Latency Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(threads)
        ax2.legend(fontsize=11)
        ax2.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(charts_dir, "thread_config_latency_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"线程配置延迟分布图表已保存到: {output_path}")
    plt.close()

def main():
    print("===== 开始生成PNG格式图表 =====")
    
    # 读取测试结果
    print("\n读取测试结果...")
    cold_start_results = read_cold_start_results()
    thread_config_results = read_thread_config_results()
    comprehensive_data = read_thread_config_comprehensive()
    
    # 生成图表
    print("\n生成冷启动相关图表...")
    generate_cold_start_factor_chart(cold_start_results)
    generate_cold_start_vs_stable_chart(cold_start_results)
    
    print("\n生成线程配置相关图表...")
    generate_thread_config_avg_latency_chart(thread_config_results)
    generate_thread_config_speedup_chart(comprehensive_data)
    generate_thread_config_memory_usage_chart(comprehensive_data)
    generate_thread_config_latency_distribution_chart(comprehensive_data)
    
    print("\n===== 所有PNG格式图表生成完成！ =====")
    print(f"图表保存位置: {charts_dir}")

if __name__ == "__main__":
    main()
