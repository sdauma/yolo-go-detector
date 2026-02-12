#!/usr/bin/env python3
# generate_cold_start_and_thread_charts.py
# 生成冷启动时间对比和线程配置性能对比图表

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体为华文中宋，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['font.family'] = ['sans-serif', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 读取冷启动测试结果
def read_cold_start_results():
    """读取冷启动测试结果"""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results")
    
    # 读取Go冷启动测试结果
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
            # 提取冷启动时间
            cold_start_match = re.search(r"冷启动时间: ([0-9.]+) ms", content)
            if cold_start_match:
                go_cold_start_time = float(cold_start_match.group(1))
            # 提取稳定状态时间
            stable_match = re.search(r"稳定状态时间: ([0-9.]+) ms", content)
            if stable_match:
                go_stable_time = float(stable_match.group(1))
    
    # 解析Python冷启动测试结果
    if os.path.exists(py_cold_start_path):
        with open(py_cold_start_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 提取冷启动时间
            cold_start_match = re.search(r"冷启动时间: ([0-9.]+) ms", content)
            if cold_start_match:
                py_cold_start_time = float(cold_start_match.group(1))
            # 提取稳定状态时间
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

# 读取线程配置测试结果
def read_thread_config_results():
    """读取线程配置测试结果"""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results")
    
    threads = [1, 2, 4, 8]
    go_results = {}
    py_results = {}
    
    for thread in threads:
        # 读取Go线程配置测试结果
        go_thread_path = os.path.join(results_dir, f"go_thread_{thread}_result.txt")
        if os.path.exists(go_thread_path):
            with open(go_thread_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 提取平均延迟
                avg_match = re.search(r"平均延迟: ([0-9.]+) ms", content)
                if avg_match:
                    go_results[thread] = float(avg_match.group(1))
        
        # 读取Python线程配置测试结果
        py_thread_path = os.path.join(results_dir, f"python_thread_{thread}_result.txt")
        if os.path.exists(py_thread_path):
            with open(py_thread_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 提取平均延迟
                avg_match = re.search(r"平均延迟: ([0-9.]+) ms", content)
                if avg_match:
                    py_results[thread] = float(avg_match.group(1))
    
    return {
        "go": go_results,
        "python": py_results
    }

# 生成冷启动时间对比图表
def generate_cold_start_chart(cold_start_results):
    """生成冷启动时间对比图表"""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results")
    output_path = os.path.join(results_dir, "cold_start_comparison.pdf")
    
    labels = ['Go', 'Python']
    cold_start_times = [cold_start_results['go']['cold_start'], cold_start_results['python']['cold_start']]
    stable_times = [cold_start_results['go']['stable'], cold_start_results['python']['stable']]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, cold_start_times, width, label='Cold Start Time')
    rects2 = ax.bar(x + width/2, stable_times, width, label='Stable Time')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (ms)')
    ax.set_title('Cold Start vs Stable Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"冷启动时间对比图表已保存到: {output_path}")

# 生成线程配置性能对比图表
def generate_thread_config_chart(thread_config_results):
    """生成线程配置性能对比图表"""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_dir = os.path.join(project_root, "results")
    output_path = os.path.join(results_dir, "thread_config_comparison.pdf")
    
    threads = list(thread_config_results['go'].keys())
    go_times = list(thread_config_results['go'].values())
    py_times = list(thread_config_results['python'].values())
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(threads, go_times, marker='o', label='Go')
    ax.plot(threads, py_times, marker='s', label='Python')
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Performance vs Number of Threads')
    ax.set_xticks(threads)
    ax.legend()
    ax.grid(linestyle="--", linewidth=0.5)
    
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"线程配置性能对比图表已保存到: {output_path}")

# 主函数
def main():
    # 读取测试结果
    cold_start_results = read_cold_start_results()
    thread_config_results = read_thread_config_results()
    
    # 生成图表
    generate_cold_start_chart(cold_start_results)
    generate_thread_config_chart(thread_config_results)
    
    print("所有图表生成完成！")

if __name__ == "__main__":
    main()
