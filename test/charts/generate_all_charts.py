# test/charts/generate_all_charts.py
# 面向核心期刊的论文图表生成脚本

import matplotlib.pyplot as plt
import numpy as np
import os
import json

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 定义路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
output_dir = os.path.join(base_dir, "results", "charts")
final_charts_dir = os.path.join(output_dir, "final_charts")

# 创建最终图表目录
os.makedirs(final_charts_dir, exist_ok=True)

# 创建图表输出目录
os.makedirs(output_dir, exist_ok=True)



print("=" * 60)
print("生成核心期刊论文图表")
print("=" * 60)

# ============================================
# 图 2: 三种并发架构吞吐量对比 (对应第 5 章)
# ============================================
def read_architecture_throughput():
    """从 go_architecture_comparison.txt 读取 12 并发的吞吐量数据"""
    throughput_data = {
        'Unsafe Shared': 0,
        'Mutex Shared': 0,
        'Session Pool': 0
    }
    
    try:
        with open(os.path.join(results_dir, 'go_architecture_comparison.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_arch = None
        for i, line in enumerate(lines):
            line_content = line.strip()
            if '===== Unsafe Shared =====' in line_content:
                current_arch = 'Unsafe Shared'
            elif '===== Mutex Shared =====' in line_content:
                current_arch = 'Mutex Shared'
            elif '===== Session Pool =====' in line_content:
                current_arch = 'Session Pool'
            elif ('并发度: 12' in line_content or '池大小: 12' in line_content) and current_arch:
                # 查找下一行的吞吐量数据
                for j in range(i+1, min(i+11, len(lines))):
                    next_line = lines[j].strip()
                    if '吞吐量:' in next_line:
                        throughput = float(next_line.split(':')[1].split()[0].strip())
                        throughput_data[current_arch] = round(throughput, 2)
                        break
        
        if throughput_data['Unsafe Shared'] == 0 or throughput_data['Mutex Shared'] == 0 or throughput_data['Session Pool'] == 0:
            raise ValueError("无法读取吞吐量数据")
        
        return [throughput_data['Unsafe Shared'], throughput_data['Mutex Shared'], throughput_data['Session Pool']]
    except Exception as e:
        print(f"读取吞吐量数据失败: {e}")
        raise

def plot_throughput_comparison():
    # 从文件读取 12 并发的吞吐量数据
    architectures = ['Unsafe\nShared', 'Mutex\nShared', 'Session\nPool']
    throughput = read_architecture_throughput()
    colors = ['#CCCCCC', '#666666', '#000000']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(architectures, throughput, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('吞吐量 (REQ/s)', fontsize=8, fontweight='bold')
    ax.set_title('三种并发架构的吞吐量对比', fontsize=9, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 计算性能提升
    improvement = throughput[2] / throughput[1]
    ax.annotate(f'Session Pool\nvs Mutex: {improvement:.0f} 倍提升', 
                xy=(2.2, throughput[2]), xytext=(2.3, throughput[2] * 1.08),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkA=0, shrinkB=6),
                fontsize=8, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_throughput_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig2_throughput_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 2 已生成：fig2_throughput_comparison.png")

plot_throughput_comparison()

# ============================================
# 图 3: 内存占用随并发数变化 (对应第 5 章)
# ============================================
def read_memory_data():
    """从文件读取内存占用数据"""
    try:
        # 读取 Python 数据
        python_memory = []
        with open(os.path.join(results_dir, 'python_architecture_comparison.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '架构=Shared' in line:
                    parts = line.split(',')
                    concurrency = int(parts[1].split('=')[1].strip())
                    memory_str = parts[-2].split('=')[1].strip()
                    memory = float(memory_str.split()[0])  # 提取数值部分
                    python_memory.append((concurrency, memory))
        # 读取 Go 数据
        go_memory = []
        with open(os.path.join(results_dir, 'go_architecture_comparison.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            in_session_pool = False
            current_pool_size = None
            
            for line in lines:
                line = line.strip()
                
                if '===== Session Pool =====' in line:
                    in_session_pool = True
                elif in_session_pool and '池大小:' in line:
                    pool_size_str = line.split(':')[1].strip()
                    try:
                        current_pool_size = int(pool_size_str)
                    except ValueError:
                        current_pool_size = None
                elif in_session_pool and '峰值RSS:' in line and current_pool_size is not None:
                    memory_str = line.split(':')[1].strip()
                    try:
                        memory = float(memory_str.split()[0])
                        go_memory.append((current_pool_size, memory))
                    except ValueError:
                        pass
                elif line.startswith('=====') and 'Session Pool' not in line:
                    in_session_pool = False
        
        # 提取共同的并发数（同时存在于 Python 和 Go 数据中）
        python_concurrencies = set([item[0] for item in python_memory])
        go_concurrencies = set([item[0] for item in go_memory])
        common_concurrencies = sorted(list(python_concurrencies.intersection(go_concurrencies)))
        
        if not common_concurrencies:
            raise ValueError("没有找到共同的并发数数据")
        
        # 提取数据
        python_values = []
        go_values = []
        
        for c in common_concurrencies:
            # 查找 Python 数据
            for item in python_memory:
                if item[0] == c:
                    python_values.append(item[1])
                    break
            
            # 查找 Go 数据
            for item in go_memory:
                if item[0] == c:
                    go_values.append(item[1])
                    break
        
        return common_concurrencies, python_values, go_values
    except Exception as e:
        print(f"读取内存数据失败: {e}")
        raise

def plot_memory_comparison():
    concurrency, python_memory_values, go_memory_values = read_memory_data()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Python 曲线
    ax.plot(concurrency, python_memory_values, 'o-', label='Python 传统部署',
            color='#666666', linewidth=2.5, markersize=8, markeredgewidth=1.5)
    # Go 曲线
    ax.plot(concurrency, go_memory_values, 's--', label='Go Session Pool 架构',
            color='#000000', linewidth=2.5, markersize=8, markeredgewidth=1.5)
    
    ax.set_xlabel('并发数', fontsize=8, fontweight='bold')
    ax.set_ylabel('内存占用 (MB)', fontsize=8, fontweight='bold')
    ax.set_title('不同并发数下的内存占用对比', fontsize=9, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加关键数据点标注
    ax.annotate('线性增长', xy=(8, python_memory_values[3]), xytext=(6, python_memory_values[3]+500),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2),
                fontsize=8, color='#666666', fontweight='bold')
    ax.annotate('几乎恒定', xy=(8, go_memory_values[3]), xytext=(5, go_memory_values[3]+50),
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2),
                fontsize=8, color='#000000', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_memory_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig3_memory_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 3 已生成：fig3_memory_comparison.png")

plot_memory_comparison()

# ============================================
# 图 4: CPU 推理批处理效应 (对应第 5 章)
# ============================================
def read_batch_data():
    """从 go_batch_inference_result.json 读取批处理测试数据"""
    try:
        with open(os.path.join(results_dir, 'go_batch_inference_result.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        batch_sizes = []
        batch_latency = []
        batch_throughput = []
        
        for result in data.get('results', []):
            batch_sizes.append(result.get('batch_size'))
            batch_latency.append(round(result.get('per_image_time_ms'), 0))
            batch_throughput.append(round(result.get('throughput_images_per_sec'), 2))
        
        if not batch_sizes:
            raise ValueError("无法读取批处理数据")
        
        return batch_sizes, batch_latency, batch_throughput
    except Exception as e:
        print(f"读取批处理数据失败: {e}")
        raise

def plot_batch_effect():
    batch_sizes, batch_latency, batch_throughput = read_batch_data()
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # 左轴 - 延迟
    color = '#000000'
    ax1.set_xlabel('批处理大小', fontsize=8, fontweight='bold')
    ax1.set_ylabel('单图延迟 (ms)', color=color, fontsize=8, fontweight='bold')
    line1 = ax1.plot(batch_sizes, batch_latency, 'o-', color=color, linewidth=2.5,
                     markersize=8, label='延迟 (左轴)')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    
    # 设置 x 轴为实际的批处理大小（离散值）
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=8)
    
    # 右轴 - 吞吐量
    ax2 = ax1.twinx()
    color = '#666666'
    ax2.set_ylabel('吞吐量 (img/s)', color=color, fontsize=8, fontweight='bold')
    bars = ax2.bar(batch_sizes, batch_throughput, color=color, alpha=0.3, label='吞吐量 (右轴)')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax2.set_ylim(1.15, 1.21)
    
    # 标题和图例
    plt.suptitle('CPU 推理场景下批处理对吞吐量的影响', fontsize=9, fontweight='bold', y=0.90)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, framealpha=0.9)
    
    # 添加结论文字
    fig.text(0.5, 0.02, '结论：CPU 场景下 Batch Size 对性能无显著影响',
             fontsize=8, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{output_dir}/fig4_batch_effect.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig4_batch_effect.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 4 已生成：fig4_batch_effect.png")

plot_batch_effect()

# ============================================
# 图 5: 模型规模与语言性能 (对应第 5 章)
# ============================================
def read_model_latency_data():
    """从测试结果文件读取模型延迟数据"""
    try:
        # 读取 Python 基准测试数据 (YOLO11x)
        python_latency_11x = 0
        with open(os.path.join(results_dir, 'python_baseline_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '平均延迟：' in line or '平均延迟:' in line:
                    latency_str = line.split('：')[1].strip() if '：' in line else line.split(':')[1].strip()
                    # 提取数值部分，去掉 ms 单位
                    python_latency_11x = float(latency_str.split(' ')[0])
                    break
        
        # 读取 Go 基准测试数据 (YOLO11x)
        go_latency_11x = 0
        with open(os.path.join(results_dir, 'go_baseline_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '平均延迟：' in line or '平均延迟:' in line:
                    latency_str = line.split('：')[1].strip() if '：' in line else line.split(':')[1].strip()
                    # 提取数值部分，去掉 ms 单位
                    go_latency_11x = float(latency_str.split(' ')[0])
                    break
        
        # 读取 Python YOLO11n 强化测试数据
        python_latency_11n = 0
        with open(os.path.join(results_dir, 'python_yolo11n_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if '轮平均延迟' in line:
                    for sep in ['：', ':']:
                        if sep in line:
                            try:
                                python_latency_11n = float(line.split(sep)[1].strip().split(' ')[0])
                                if python_latency_11n > 0:
                                    break
                            except ValueError:
                                pass
                    if python_latency_11n > 0:
                        break
        
        # 读取 Go YOLO11n 强化测试数据
        go_latency_11n = 0
        with open(os.path.join(results_dir, 'go_yolo11n_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if '轮平均延迟' in line:
                    for sep in ['：', ':']:
                        if sep in line:
                            try:
                                go_latency_11n = float(line.split(sep)[1].strip().split(' ')[0])
                                if go_latency_11n > 0:
                                    break
                            except ValueError:
                                pass
                    if go_latency_11n > 0:
                        break
        
        # 检查是否读取到所有数据
        if python_latency_11x == 0:
            raise ValueError("无法读取 Python YOLO11x 模型延迟数据")
        
        if go_latency_11x == 0:
            raise ValueError("无法读取 Go YOLO11x 模型延迟数据")
        
        if python_latency_11n == 0:
            raise ValueError("无法读取 Python YOLO11n 模型延迟数据")
        
        if go_latency_11n == 0:
            raise ValueError("无法读取 Go YOLO11n 模型延迟数据")
        
        return [python_latency_11n, python_latency_11x], [go_latency_11n, go_latency_11x]
    except Exception as e:
        print(f"读取模型延迟数据失败: {e}")
        raise

def plot_model_size_comparison():
    models = ['YOLO11n\n(轻量)', 'YOLO11x\n(大模型)']
    python_latency_values, go_latency_values = read_model_latency_data()
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars1 = ax.bar(x - width/2, python_latency_values, width, label='Python',
                   color='#666666', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, go_latency_values, width, label='Go',
                   color='#000000', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('推理延迟 (ms)', fontsize=8, fontweight='bold')
    ax.set_title('不同模型规模下的推理延迟对比', fontsize=9, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(['Python', 'Go'], fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 添加性能差异标注
    diff_percent = [(python_latency_values[0]-go_latency_values[0])/go_latency_values[0]*100,
                    (go_latency_values[1]-python_latency_values[1])/python_latency_values[1]*100]
    for i, diff in enumerate(diff_percent):
        y_pos = max(python_latency_values[i], go_latency_values[i]) + 12
        ax.text(i, y_pos, f'{diff:.1f}% 差异',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_model_size_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig5_model_size_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 5 已生成：fig5_model_size_comparison.png")

plot_model_size_comparison()

# ============================================
# 图 6: CPU 利用率对比 (对应第 5 章)
# ============================================
def read_cpu_utilization_data():
    """从测试结果文件读取 CPU 利用率数据"""
    try:
        # 读取 Go CPU 监控数据
        go_cpu_util = 0
        with open(os.path.join(results_dir, 'go_cpu_monitoring_result.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if item.get('scenario') == 'Concurrent_10x50':
                        go_cpu_util = round(item.get('avg_cpu_percent', 0), 2)
                        break
            elif isinstance(data, dict):
                go_cpu_util = round(data.get('avg_cpu_percent', 0), 2)
        
        # 读取 Python CPU 监控数据
        python_cpu_util = 0
        with open(os.path.join(results_dir, 'python_cpu_monitoring_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '平均 CPU' in line or 'Average CPU' in line:
                    for sep in ['：', ':']:
                        if sep in line:
                            python_cpu_util = float(line.split(sep)[1].strip().replace('%', ''))
                            break
                    if python_cpu_util > 0:
                        break
        
        # 检查是否读取到数据
        if go_cpu_util == 0:
            raise ValueError("无法读取 Go CPU 利用率数据")
        
        if python_cpu_util == 0:
            raise ValueError("无法读取 Python CPU 利用率数据")
        
        return [python_cpu_util, go_cpu_util]
    except Exception as e:
        print(f"读取 CPU 利用率数据失败：{e}")
        raise

def plot_cpu_utilization():
    schemes = ['Python 单实例', 'Go Session Pool (10并发)']
    cpu_util = read_cpu_utilization_data()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(schemes, cpu_util, color=['#666666', '#000000'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('CPU 利用率 (%)', fontsize=8, fontweight='bold')
    ax.set_title('不同部署方案的 CPU 利用率对比', fontsize=9, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加说明
    ax.annotate('单核利用', xy=(0.2, cpu_util[0]), xytext=(0.25, cpu_util[0] * 1.20),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2),
                fontsize=8, color='#666666', fontweight='bold')
    ax.annotate('多核并行(~6核)', xy=(1.07, cpu_util[1] * 1.01), xytext=(1.14, cpu_util[1] * 1.01),
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2),
                fontsize=8, color='#000000', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{output_dir}/fig6_cpu_utilization.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig6_cpu_utilization.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 6 已生成：fig6_cpu_utilization.png")

plot_cpu_utilization()

# ============================================
# 图 7: 长时间稳定性对比 (对应第 5 章)
# ============================================
def read_stability_data():
    """从测试结果文件读取稳定性数据（包括实际测试时长）"""
    try:
        python_duration_min = 10.0
        python_initial_rss = 0
        python_final_rss = 0
        with open(os.path.join(results_dir, 'python_long_stability_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '测试时长' in line:
                    for sep in ['：', ':']:
                        if sep in line:
                            duration_str = line.split(sep)[1].strip()
                            if 'm' in duration_str:
                                parts = duration_str.replace('s', '').split('m')
                                python_duration_min = float(parts[0]) + float(parts[1]) / 60.0
                            else:
                                python_duration_min = float(duration_str.replace('秒', '').replace('s', '')) / 60.0
                            break
                elif '初始 RSS:' in line or 'Start RSS:' in line:
                    for sep in [':', '：']:
                        if sep in line:
                            try:
                                python_initial_rss = float(line.split(sep)[1].strip().split(' ')[0])
                                break
                            except ValueError:
                                pass
                elif '最终 RSS:' in line or 'End RSS:' in line or '结束 RSS:' in line:
                    for sep in [':', '：']:
                        if sep in line:
                            try:
                                python_final_rss = float(line.split(sep)[1].strip().split(' ')[0])
                                break
                            except ValueError:
                                pass

        go_duration_min = 10.0
        go_initial_rss = 0
        go_final_rss = 0
        with open(os.path.join(results_dir, 'go_long_stability_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '测试时长' in line:
                    for sep in ['：', ':']:
                        if sep in line:
                            duration_str = line.split(sep)[1].strip()
                            if 'm' in duration_str:
                                parts = duration_str.replace('s', '').split('m')
                                go_duration_min = float(parts[0]) + float(parts[1]) / 60.0
                            else:
                                go_duration_min = float(duration_str.replace('秒', '').replace('s', '')) / 60.0
                            break
                elif '初始 RSS:' in line or 'Start RSS:' in line:
                    for sep in [':', '：']:
                        if sep in line:
                            try:
                                go_initial_rss = float(line.split(sep)[1].strip().split(' ')[0])
                                break
                            except ValueError:
                                pass
                elif '最终 RSS:' in line or 'End RSS:' in line or '结束 RSS:' in line:
                    for sep in [':', '：']:
                        if sep in line:
                            try:
                                go_final_rss = float(line.split(sep)[1].strip().split(' ')[0])
                                break
                            except ValueError:
                                pass

        if python_initial_rss == 0 or python_final_rss == 0:
            raise ValueError("无法读取 Python 稳定性数据")

        if go_initial_rss == 0 or go_final_rss == 0:
            raise ValueError("无法读取 Go 稳定性数据")

        actual_duration = max(python_duration_min, go_duration_min)
        return python_initial_rss, python_final_rss, go_initial_rss, go_final_rss, actual_duration
    except Exception as e:
        print(f"读取稳定性数据失败：{e}")
        raise

def plot_stability():
    python_initial_rss, python_final_rss, go_initial_rss, go_final_rss, actual_duration = read_stability_data()

    stability_time = np.linspace(0, actual_duration, int(actual_duration * 6) + 1)

    python_drift_rate = (python_final_rss - python_initial_rss) / actual_duration
    go_drift_rate = (go_final_rss - go_initial_rss) / actual_duration

    python_rss = python_initial_rss + python_drift_rate * stability_time
    go_rss = go_initial_rss + go_drift_rate * stability_time

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(stability_time, python_rss, 'o-', label='Python',
            color='#666666', linewidth=2, markersize=5, markevery=10)
    ax.plot(stability_time, go_rss, 's--', label='Go Session Pool',
            color='#000000', linewidth=2, markersize=5, markevery=10)

    ax.set_xlabel('运行时间 (分钟)', fontsize=8, fontweight='bold')
    ax.set_ylabel('RSS 内存占用 (MB)', fontsize=8, fontweight='bold')
    ax.set_title(f'长时间运行的内存漂移对比（{actual_duration:.0f} 分钟测试）', fontsize=9, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 0.95), framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    ax.annotate(f'Python 漂移: +{(python_rss[-1]-python_rss[0]):.2f} MB',
                xy=(stability_time[-1], python_rss[-1]),
                xytext=(stability_time[-1] * 0.7, python_rss[-1] - 30),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2),
                fontsize=8, color='#666666', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))

    ax.annotate(f'Go 漂移: +{(go_rss[-1]-go_rss[0]):.2f} MB',
                xy=(stability_time[-1], go_rss[-1]),
                xytext=(stability_time[-1] * 0.7, go_rss[-1] + 30),
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2),
                fontsize=8, color='#000000', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_stability.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig7_stability.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"图 7 已生成：fig7_stability.png（实际测试时长: {actual_duration:.1f} 分钟）")

plot_stability()

print("\n" + "=" * 60)
print("所有 7 张图表已生成完毕！")
print("=" * 60)
print(f"\n图表保存位置：{os.path.abspath(output_dir)}")
print("\n生成的图表列表:")
print("  1. fig2_throughput_comparison.png   - 三种并发架构的吞吐量对比")
print("  2. fig3_memory_comparison.png       - 不同并发数下的内存占用对比")
print("  3. fig4_batch_effect.png            - CPU 推理场景下批处理对吞吐量的影响")
print("  4. fig5_model_size_comparison.png   - 不同模型规模下的推理延迟对比")
print("  5. fig6_cpu_utilization.png         - 不同部署方案的 CPU 利用率对比")
print("  6. fig7_stability.png               - 长时间运行的内存漂移对比")
print("\n提示：这些图表可以直接插入到 LaTeX 论文中")
print("=" * 60)
