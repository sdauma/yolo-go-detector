#!/usr/bin/env python3
# generate_reinforced_charts.py
# 生成强化实验的图表文件

import os
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*iCCP.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import re

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 其他字体参数设置
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['patch.linewidth'] = 0.8

# 获取项目根目录和图表保存目录
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results")
charts_dir = os.path.join(results_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

# ========== 数据读取函数 ==========
def read_latency_data(file_path):
    """从延迟数据文件读取数据"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        times = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('====='):
                    try:
                        times.append(float(line))
                    except ValueError:
                        continue
        
        if not times:
            raise ValueError(f"文件 {file_path} 中没有有效延迟数据")
        
        return times
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"读取延迟数据失败：{e}")

def read_result_file(file_path):
    """从结果文件读取指标"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        result = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取平均延迟
            avg_match = re.search(r'平均延迟:\s*([\d.]+)\s*ms', content)
            if avg_match:
                result['avg'] = float(avg_match.group(1))
            
            # 提取标准差
            std_match = re.search(r'标准差:\s*([\d.]+)\s*ms', content)
            if std_match:
                result['std'] = float(std_match.group(1))
            
            # 提取P50延迟
            p50_match = re.search(r'P50延迟:\s*([\d.]+)\s*ms', content)
            if p50_match:
                result['p50'] = float(p50_match.group(1))
            
            # 提取P90延迟
            p90_match = re.search(r'P90延迟:\s*([\d.]+)\s*ms', content)
            if p90_match:
                result['p90'] = float(p90_match.group(1))
            
            # 提取P95延迟
            p95_match = re.search(r'P95延迟:\s*([\d.]+)\s*ms', content)
            if p95_match:
                result['p95'] = float(p95_match.group(1))
            
            # 提取内存增加量
            mem_match = re.search(r'内存增加量:\s*([\d.]+)\s*MB', content)
            if mem_match:
                result['memory_increase'] = float(mem_match.group(1))
        
        if not result:
            raise ValueError(f"文件 {file_path} 中没有有效结果数据")
        
        return result
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"读取结果文件失败：{e}")

def read_cold_start_detailed_log(file_path):
    """从冷启动详细日志读取数据"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        data = {
            'session_creation': [],
            'model_loading': [],
            'first_inference': [],
            'total_cold_start': []
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 提取会话创建时间
                session_match = re.search(r'会话创建时间:\s*([\d.]+)\s*ms', line)
                if session_match:
                    data['session_creation'].append(float(session_match.group(1)))
                
                # 提取模型加载时间
                model_match = re.search(r'模型加载时间:\s*([\d.]+)\s*ms', line)
                if model_match:
                    data['model_loading'].append(float(model_match.group(1)))
                
                # 提取首次推理时间
                inference_match = re.search(r'首次推理时间:\s*([\d.]+)\s*ms', line)
                if inference_match:
                    data['first_inference'].append(float(inference_match.group(1)))
                
                # 提取总冷启动时间
                total_match = re.search(r'总冷启动时间:\s*([\d.]+)\s*ms', line)
                if total_match:
                    data['total_cold_start'].append(float(total_match.group(1)))
        
        if not any(data.values()):
            raise ValueError(f"文件 {file_path} 中没有有效冷启动数据")
        
        return data
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"读取冷启动数据失败：{e}")

def read_memory_detailed_log(file_path):
    """从内存详细日志读取数据"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        data = {
            'interpreter_memory': [],
            'model_loaded_memory': [],
            'post_inference_memory': [],
            'memory_increase': []
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 提取解释器常驻内存
                interpreter_match = re.search(r'解释器常驻内存:\s*([\d.]+)\s*MB', line)
                if interpreter_match:
                    data['interpreter_memory'].append(float(interpreter_match.group(1)))
                
                # 提取模型加载后内存
                model_match = re.search(r'模型加载后内存:\s*([\d.]+)\s*MB', line)
                if model_match:
                    data['model_loaded_memory'].append(float(model_match.group(1)))
                
                # 提取推理后内存
                inference_match = re.search(r'推理后内存:\s*([\d.]+)\s*MB', line)
                if inference_match:
                    data['post_inference_memory'].append(float(inference_match.group(1)))
                
                # 提取内存增加量
                increase_match = re.search(r'内存增加量:\s*([\d.]+)\s*MB', line)
                if increase_match:
                    data['memory_increase'].append(float(increase_match.group(1)))
        
        if not any(data.values()):
            raise ValueError(f"文件 {file_path} 中没有有效内存数据")
        
        return data
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"读取内存数据失败：{e}")

def read_detections(file_path):
    """从检测结果文件读取bounding boxes"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        boxes = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 提取bounding box信息
                box_match = re.search(r'目标 \d+: 类别=(\d+), 置信度=([\d.]+), 坐标=\(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)', line)
                if box_match:
                    boxes.append({
                        'class_id': int(box_match.group(1)),
                        'confidence': float(box_match.group(2)),
                        'x': float(box_match.group(3)),
                        'y': float(box_match.group(4)),
                        'width': float(box_match.group(5)),
                        'height': float(box_match.group(6))
                    })
        
        if not boxes:
            raise ValueError(f"文件 {file_path} 中没有有效检测数据")
        
        return boxes
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"读取检测数据失败：{e}")

def calculate_l2_error(boxes1, boxes2):
    """计算两组bounding boxes的L2误差"""
    if len(boxes1) != len(boxes2):
        return float('inf')
    
    total_error = 0
    for b1, b2 in zip(boxes1, boxes2):
        # 计算坐标差异
        dx = b1['x'] - b2['x']
        dy = b1['y'] - b2['y']
        dw = b1['width'] - b2['width']
        dh = b1['height'] - b2['height']
        
        # 计算L2误差
        error = np.sqrt(dx**2 + dy**2 + dw**2 + dh**2)
        total_error += error
    
    return total_error / len(boxes1) if boxes1 else 0

# ========== 图表生成函数 ==========
def generate_reinforced_performance_comparison():
    """生成强化性能测试对比图表"""
    try:
        py_large_result = read_result_file(os.path.join(results_dir, "python_reinforced_result.txt"))
    except Exception:
        py_large_result = {}
    try:
        go_large_result = read_result_file(os.path.join(results_dir, "go_reinforced_result.txt"))
    except Exception:
        go_large_result = {}
    try:
        py_small_result = read_result_file(os.path.join(results_dir, "python_reinforced_small_result.txt"))
    except Exception:
        py_small_result = {}
    try:
        go_small_result = read_result_file(os.path.join(results_dir, "go_reinforced_small_result.txt"))
    except Exception:
        go_small_result = {}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 大模型对比
    if py_large_result and go_large_result:
        metrics = ['平均延迟', 'P50延迟', 'P90延迟', 'P95延迟']
        py_values = [py_large_result.get('avg', 0), py_large_result.get('p50', 0), 
                     py_large_result.get('p90', 0), py_large_result.get('p95', 0)]
        go_values = [go_large_result.get('avg', 0), go_large_result.get('p50', 0), 
                     go_large_result.get('p90', 0), go_large_result.get('p95', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, py_values, width, label='Python', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, go_values, width, label='Go', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('性能指标', fontsize=11)
        ax1.set_ylabel('延迟 (ms)', fontsize=11)
        ax1.set_title('大模型 (YOLO11x) 性能对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 轻模型对比
    if py_small_result and go_small_result:
        metrics = ['平均延迟', 'P50延迟', 'P90延迟', 'P95延迟']
        py_values = [py_small_result.get('avg', 0), py_small_result.get('p50', 0), 
                     py_small_result.get('p90', 0), py_small_result.get('p95', 0)]
        go_values = [go_small_result.get('avg', 0), go_small_result.get('p50', 0), 
                     go_small_result.get('p90', 0), go_small_result.get('p95', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, py_values, width, label='Python', color='#FF6B6B', alpha=0.8)
        bars2 = ax2.bar(x + width/2, go_values, width, label='Go', color='#4ECDC4', alpha=0.8)
        
        ax2.set_xlabel('性能指标', fontsize=11)
        ax2.set_ylabel('延迟 (ms)', fontsize=11)
        ax2.set_title('轻模型 (YOLO11n) 性能对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path_png = os.path.join(charts_dir, "reinforced_performance_comparison.png")
    output_path_pdf = os.path.join(charts_dir, "reinforced_performance_comparison.pdf")
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"强化性能对比图表已保存到: {output_path_png}")
    print(f"强化性能对比图表已保存到: {output_path_pdf}")
    plt.close()

def generate_ttest_visualization():
    """生成t-test结果可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 读取延迟数据（容错）
    try:
        py_large_times = read_latency_data(os.path.join(results_dir, "python_reinforced_12threads_latency_data.txt"))
    except Exception:
        py_large_times = []
    try:
        go_large_times = read_latency_data(os.path.join(results_dir, "go_reinforced_latency_data.txt"))
    except Exception:
        go_large_times = []
    try:
        py_small_times = read_latency_data(os.path.join(results_dir, "python_reinforced_small_latency_data.txt"))
    except Exception:
        py_small_times = []
    try:
        go_small_times = read_latency_data(os.path.join(results_dir, "go_reinforced_small_latency_data.txt"))
    except Exception:
        go_small_times = []
    
    # 大模型延迟分布
    ax1 = axes[0, 0]
    if py_large_times and go_large_times:
        ax1.hist(py_large_times, bins=30, alpha=0.6, label='Python', color='#FF6B6B', density=True)
        ax1.hist(go_large_times, bins=30, alpha=0.6, label='Go', color='#4ECDC4', density=True)
        ax1.set_xlabel('延迟 (ms)', fontsize=10)
        ax1.set_ylabel('密度', fontsize=10)
        ax1.set_title('大模型 (YOLO11x) 延迟分布', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 计算t-test
        t_stat, p_value = stats.ttest_ind(py_large_times, go_large_times)
        ax1.text(0.95, 0.95, f't-statistic: {t_stat:.3f}\np-value: {p_value:.3e}', 
                transform=ax1.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 轻模型延迟分布
    ax2 = axes[0, 1]
    if py_small_times and go_small_times:
        ax2.hist(py_small_times, bins=30, alpha=0.6, label='Python', color='#FF6B6B', density=True)
        ax2.hist(go_small_times, bins=30, alpha=0.6, label='Go', color='#4ECDC4', density=True)
        ax2.set_xlabel('延迟 (ms)', fontsize=10)
        ax2.set_ylabel('密度', fontsize=10)
        ax2.set_title('轻模型 (YOLO11n) 延迟分布', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 计算t-test
        t_stat, p_value = stats.ttest_ind(py_small_times, go_small_times)
        ax2.text(0.95, 0.95, f't-statistic: {t_stat:.3f}\np-value: {p_value:.3e}', 
                transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 大模型箱线图
    ax3 = axes[1, 0]
    if py_large_times and go_large_times:
        data = [py_large_times, go_large_times]
        bp = ax3.boxplot(data, tick_labels=['Python', 'Go'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_alpha(0.6)
        ax3.set_ylabel('延迟 (ms)', fontsize=10)
        ax3.set_title('大模型 (YOLO11x) 延迟箱线图', fontsize=11, fontweight='bold')
        ax3.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 轻模型箱线图
    ax4 = axes[1, 1]
    if py_small_times and go_small_times:
        data = [py_small_times, go_small_times]
        bp = ax4.boxplot(data, tick_labels=['Python', 'Go'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_alpha(0.6)
        ax4.set_ylabel('延迟 (ms)', fontsize=10)
        ax4.set_title('轻模型 (YOLO11n) 延迟箱线图', fontsize=11, fontweight='bold')
        ax4.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path_png = os.path.join(charts_dir, "reinforced_ttest_visualization.png")
    output_path_pdf = os.path.join(charts_dir, "reinforced_ttest_visualization.pdf")
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"t-test可视化图表已保存到: {output_path_png}")
    print(f"t-test可视化图表已保存到: {output_path_pdf}")
    plt.close()

def generate_cold_start_decomposition():
    """生成冷启动分解测试图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 读取冷启动数据（容错）
    try:
        py_large_cold = read_cold_start_detailed_log(os.path.join(results_dir, "python_cold_start_detailed_log.txt"))
    except Exception:
        py_large_cold = {}
    try:
        go_large_cold = read_cold_start_detailed_log(os.path.join(results_dir, "go_cold_start_detailed_log.txt"))
    except Exception:
        try:
            go_large_cold = read_cold_start_detailed_log(os.path.join(results_dir, "go_cold_start_decomposition_log.txt"))
        except Exception:
            go_large_cold = {}
    
    # 大模型冷启动分解
    ax1 = axes[0, 0]
    if py_large_cold and go_large_cold:
        categories = ['会话创建', '模型加载', '首次推理', '总冷启动']
        # 检查数据是否为空并计算均值
        def safe_mean(data):
            if data and len(data) > 0:
                return np.mean(data)
            return 0
        
        py_means = [
            safe_mean(py_large_cold.get('session_creation', [])),
            safe_mean(py_large_cold.get('model_loading', [])),
            safe_mean(py_large_cold.get('first_inference', [])),
            safe_mean(py_large_cold.get('total_cold_start', []))
        ]
        go_means = [
            safe_mean(go_large_cold.get('session_creation', [])),
            safe_mean(go_large_cold.get('model_loading', [])),
            safe_mean(go_large_cold.get('first_inference', [])),
            safe_mean(go_large_cold.get('total_cold_start', []))
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, py_means, width, label='Python', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, go_means, width, label='Go', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('冷启动阶段', fontsize=10)
        ax1.set_ylabel('时间 (ms)', fontsize=10)
        ax1.set_title('大模型 (YOLO11x) 冷启动分解', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 大模型冷启动因子分析
    ax2 = axes[0, 1]
    if py_large_cold and go_large_cold:
        py_total = safe_mean(py_large_cold.get('total_cold_start', []))
        go_total = safe_mean(go_large_cold.get('total_cold_start', []))
        
        try:
            py_stable = read_result_file(os.path.join(results_dir, "python_reinforced_result.txt"))
        except Exception:
            py_stable = {}
        try:
            go_stable = read_result_file(os.path.join(results_dir, "go_reinforced_result.txt"))
        except Exception:
            go_stable = {}
        
        if py_stable and go_stable:
            py_stable_latency = py_stable.get('avg', 0)
            go_stable_latency = go_stable.get('avg', 0)
            
            py_factor = py_total / py_stable_latency if py_stable_latency > 0 else 0
            go_factor = go_total / go_stable_latency if go_stable_latency > 0 else 0
            
            languages = ['Python', 'Go']
            factors = [py_factor, go_factor]
            
            bars = ax2.bar(languages, factors, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax2.set_ylabel('冷启动因子', fontsize=10)
            ax2.set_title('大模型 (YOLO11x) 冷启动因子分析', fontsize=11, fontweight='bold')
            ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
            
            # 添加数值标签
            for bar, factor in zip(bars, factors):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{factor:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # 大模型冷启动时间分布
    ax3 = axes[1, 0]
    if py_large_cold and go_large_cold:
        ax3.hist(py_large_cold['total_cold_start'], bins=20, alpha=0.6, label='Python', color='#FF6B6B', density=True)
        ax3.hist(go_large_cold['total_cold_start'], bins=20, alpha=0.6, label='Go', color='#4ECDC4', density=True)
        ax3.set_xlabel('总冷启动时间 (ms)', fontsize=10)
        ax3.set_ylabel('密度', fontsize=10)
        ax3.set_title('大模型 (YOLO11x) 冷启动时间分布', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 大模型冷启动箱线图
    ax4 = axes[1, 1]
    if py_large_cold and go_large_cold:
        data = [py_large_cold['total_cold_start'], go_large_cold['total_cold_start']]
        bp = ax4.boxplot(data, tick_labels=['Python', 'Go'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_alpha(0.6)
        ax4.set_ylabel('总冷启动时间 (ms)', fontsize=10)
        ax4.set_title('大模型 (YOLO11x) 冷启动时间箱线图', fontsize=11, fontweight='bold')
        ax4.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path_png = os.path.join(charts_dir, "reinforced_cold_start_decomposition.png")
    output_path_pdf = os.path.join(charts_dir, "reinforced_cold_start_decomposition.pdf")
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"冷启动分解图表已保存到: {output_path_png}")
    print(f"冷启动分解图表已保存到: {output_path_pdf}")
    plt.close()

def generate_memory_comparison():
    """生成内存使用对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 读取内存数据（容错）
    try:
        py_large_memory = read_memory_detailed_log(os.path.join(results_dir, "python_memory_detailed_log.txt"))
    except Exception:
        py_large_memory = {}
    try:
        go_large_memory = read_memory_detailed_log(os.path.join(results_dir, "go_memory_detailed_log.txt"))
    except Exception:
        go_large_memory = {}
    
    # 大模型内存对比
    ax1 = axes[0, 0]
    if py_large_memory and go_large_memory:
        categories = ['解释器常驻', '模型加载后', '推理后']
        py_means = [
            np.mean(py_large_memory['interpreter_memory']),
            np.mean(py_large_memory['model_loaded_memory']),
            np.mean(py_large_memory['post_inference_memory'])
        ]
        go_means = [
            np.mean(go_large_memory['interpreter_memory']),
            np.mean(go_large_memory['model_loaded_memory']),
            np.mean(go_large_memory['post_inference_memory'])
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, py_means, width, label='Python', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, go_means, width, label='Go', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('内存阶段', fontsize=10)
        ax1.set_ylabel('内存使用 (MB)', fontsize=10)
        ax1.set_title('大模型 (YOLO11x) 内存使用对比', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 大模型内存增加量对比
    ax2 = axes[0, 1]
    if py_large_memory and go_large_memory:
        py_increase = np.mean(py_large_memory['memory_increase'])
        go_increase = np.mean(go_large_memory['memory_increase'])
        
        languages = ['Python', 'Go']
        increases = [py_increase, go_increase]
        
        bars = ax2.bar(languages, increases, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax2.set_ylabel('内存增加量 (MB)', fontsize=10)
        ax2.set_title('大模型 (YOLO11x) 内存增加量对比', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar, increase in zip(bars, increases):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 大模型内存使用分布
    ax3 = axes[1, 0]
    if py_large_memory and go_large_memory:
        ax3.hist(py_large_memory['post_inference_memory'], bins=20, alpha=0.6, 
                label='Python', color='#FF6B6B', density=True)
        ax3.hist(go_large_memory['post_inference_memory'], bins=20, alpha=0.6, 
                label='Go', color='#4ECDC4', density=True)
        ax3.set_xlabel('推理后内存使用 (MB)', fontsize=10)
        ax3.set_ylabel('密度', fontsize=10)
        ax3.set_title('大模型 (YOLO11x) 内存使用分布', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 大模型内存使用箱线图
    ax4 = axes[1, 1]
    if py_large_memory and go_large_memory:
        data = [py_large_memory['post_inference_memory'], go_large_memory['post_inference_memory']]
        bp = ax4.boxplot(data, tick_labels=['Python', 'Go'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_alpha(0.6)
        ax4.set_ylabel('推理后内存使用 (MB)', fontsize=10)
        ax4.set_title('大模型 (YOLO11x) 内存使用箱线图', fontsize=11, fontweight='bold')
        ax4.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    output_path_png = os.path.join(charts_dir, "reinforced_memory_comparison.png")
    output_path_pdf = os.path.join(charts_dir, "reinforced_memory_comparison.pdf")
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"内存对比图表已保存到: {output_path_png}")
    print(f"内存对比图表已保存到: {output_path_pdf}")
    plt.close()

def generate_output_consistency():
    """生成输出一致性验证图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 读取检测结果（容错）
    try:
        py_large_boxes = read_detections(os.path.join(results_dir, "python_yolo11x_detections.txt"))
    except Exception:
        py_large_boxes = []
    try:
        go_large_boxes = read_detections(os.path.join(results_dir, "go_yolo11x_detections.txt"))
    except Exception:
        go_large_boxes = []
    try:
        py_small_boxes = read_detections(os.path.join(results_dir, "python_yolo11n_detections.txt"))
    except Exception:
        py_small_boxes = []
    try:
        go_small_boxes = read_detections(os.path.join(results_dir, "go_yolo11n_detections.txt"))
    except Exception:
        go_small_boxes = []
    
    # 大模型检测数量对比
    ax1 = axes[0, 0]
    py_large_count = len(py_large_boxes)
    go_large_count = len(go_large_boxes)
    
    languages = ['Python', 'Go']
    counts = [py_large_count, go_large_count]
    
    bars = ax1.bar(languages, counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_ylabel('检测目标数量', fontsize=10)
    ax1.set_title('大模型 (YOLO11x) 检测目标数量对比', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 大模型L2误差
    ax2 = axes[0, 1]
    if py_large_boxes and go_large_boxes:
        l2_error = calculate_l2_error(py_large_boxes, go_large_boxes)
        
        languages = ['Python vs Go']
        errors = [l2_error]
        
        bars = ax2.bar(languages, errors, color='#9B59B6', alpha=0.8)
        ax2.set_ylabel('L2误差', fontsize=10)
        ax2.set_title('大模型 (YOLO11x) Bounding Box L2误差', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 轻模型检测数量对比
    ax3 = axes[1, 0]
    py_small_count = len(py_small_boxes)
    go_small_count = len(go_small_boxes)
    
    counts = [py_small_count, go_small_count]
    
    bars = ax3.bar(languages, counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax3.set_ylabel('检测目标数量', fontsize=10)
    ax3.set_title('轻模型 (YOLO11n) 检测目标数量对比', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 轻模型L2误差
    ax4 = axes[1, 1]
    if py_small_boxes and go_small_boxes:
        l2_error = calculate_l2_error(py_small_boxes, go_small_boxes)
        
        errors = [l2_error]
        
        bars = ax4.bar(languages, errors, color='#9B59B6', alpha=0.8)
        ax4.set_ylabel('L2误差', fontsize=10)
        ax4.set_title('轻模型 (YOLO11n) Bounding Box L2误差', fontsize=11, fontweight='bold')
        ax4.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 添加数值标签
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path_png = os.path.join(charts_dir, "reinforced_output_consistency.png")
    output_path_pdf = os.path.join(charts_dir, "reinforced_output_consistency.pdf")
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"输出一致性图表已保存到: {output_path_png}")
    print(f"输出一致性图表已保存到: {output_path_pdf}")
    plt.close()

# ========== 主函数 ==========
def main():
    print("===== 开始生成强化实验图表 =====\n")
    failed = []
    
    chart_funcs = [
        ("强化性能对比图表", generate_reinforced_performance_comparison),
        ("t-test可视化图表", generate_ttest_visualization),
        ("冷启动分解图表", generate_cold_start_decomposition),
        ("内存对比图表", generate_memory_comparison),
        ("输出一致性图表", generate_output_consistency),
    ]
    
    for name, func in chart_funcs:
        print(f"\n生成{name}...")
        try:
            func()
        except Exception as e:
            print(f"  [WARN] {name} 生成失败: {e}")
            failed.append(name)
    
    print("\n===== 强化实验图表生成完成 =====")
    if failed:
        print(f"以下图表生成失败: {', '.join(failed)}")
    print(f"所有图表已保存到: {charts_dir}")

if __name__ == "__main__":
    main()
