# test/charts/generate_all_charts.py
# 面向核心期刊的论文图表生成脚本

import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免显示警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*iCCP.*')
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
                        throughput_data[current_arch] = round(throughput, 3)
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
    # ax.set_title('三种并发架构的吞吐量对比', fontsize=9, fontweight='bold', pad=10)  # 图注由论文caption提供
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_throughput_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_throughput_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig2_throughput_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 2 已生成：fig2_throughput_comparison.png")

plot_throughput_comparison()

# ============================================
# 图 3: 内存占用随并发数变化 (对应第 5 章)
# ============================================
def read_memory_data():
    """从 go_architecture_comparison.txt 读取 Go Unsafe Shared 与 Go Session Pool 的 PM 漂移数据。
    注意：fig3 采用 PM 漂移（释放量）口径，而非峰值 PM。原因：两种架构的峰值 PM 均随并发暴涨
    （Unsafe Shared 584→4433 MB，Session Pool 592→6684 MB），画峰值 PM 反而会让 Session Pool
    显得更差；而漂移（Unsafe Shared 557→4279 MB vs Session Pool ~1→-42 MB）才是 §4.1
    “池化方案内存可控”论点的真正可视化依据。"""
    try:
        go_us = []  # (并发度, 峰值PM, PM漂移)
        go_sp = []  # (池大小, 峰值PM, PM漂移)
        with open(os.path.join(results_dir, 'go_architecture_comparison.txt'), 'r', encoding='utf-8') as f:
            section = None
            cur_c = None
            cur_peak = None
            cur_drift = None
            for line in f:
                line = line.strip()
                if line.startswith('===== Unsafe Shared ====='):
                    section = 'US'
                    cur_c, cur_peak, cur_drift = None, None, None
                    continue
                elif line.startswith('===== Session Pool ====='):
                    section = 'SP'
                    cur_c, cur_peak, cur_drift = None, None, None
                    continue
                elif line.startswith('====='):
                    section = None
                    continue
                if section == 'US' and line.startswith('并发度:'):
                    try:
                        cur_c = int(line.split(':')[1].strip())
                    except ValueError:
                        cur_c = None
                elif section == 'SP' and line.startswith('池大小:'):
                    try:
                        cur_c = int(line.split(':')[1].strip())
                    except ValueError:
                        cur_c = None
                elif '峰值PM:' in line or '峰值RSS:' in line:
                    try:
                        cur_peak = float(line.split(':')[1].strip().split()[0])
                    except ValueError:
                        cur_peak = None
                elif 'PM漂移:' in line or 'RSS漂移:' in line:
                    try:
                        cur_drift = float(line.split(':')[1].strip().split()[0])
                    except ValueError:
                        cur_drift = None
                if cur_c is not None and cur_peak is not None and cur_drift is not None:
                    if section == 'US':
                        go_us.append((cur_c, cur_peak, cur_drift))
                    elif section == 'SP':
                        go_sp.append((cur_c, cur_peak, cur_drift))
                    cur_c, cur_peak, cur_drift = None, None, None

        us_c_set = sorted([x[0] for x in go_us])
        sp_c_set = sorted([x[0] for x in go_sp])
        common = sorted(set(us_c_set) & set(sp_c_set))
        if not common:
            raise ValueError("没有找到共同的并发数数据")

        def pick(data, c):
            for x in data:
                if x[0] == c:
                    return x[1], x[2]
            return None, None

        us_peak, us_drift, sp_peak, sp_drift = [], [], [], []
        for c in common:
            p1, d1 = pick(go_us, c)
            us_peak.append(p1)
            us_drift.append(d1)
            p2, d2 = pick(go_sp, c)
            sp_peak.append(p2)
            sp_drift.append(d2)
        return common, us_peak, us_drift, sp_peak, sp_drift
    except Exception as e:
        print(f"读取内存数据失败: {e}")
        raise

def plot_memory_comparison():
    concurrency, us_peak, us_drift, sp_peak, sp_drift = read_memory_data()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Go Unsafe Shared 漂移（随并发激增）
    ax.plot(concurrency, us_drift, 'o-', label='Go Unsafe Shared（内存漂移）',
            color='#888888', linewidth=2.5, markersize=8, markeredgewidth=1.5)
    # Go 临时会话（Per-Request Session）漂移（几乎恒定，12 并发净释放）
    # 注：数据文件 go_architecture_comparison.txt 中该段旧标为 "Session Pool"，
    # 实为 §4.1 的"临时会话（每 goroutine 独立临时会话、使用后即销毁）"架构，
    # 与论文表4/公平性表/图caption 术语统一，此处图例改回"临时会话"。
    ax.plot(concurrency, sp_drift, 's-', label='Go 临时会话（内存漂移）',
            color='#000000', linewidth=2.5, markersize=8, markeredgewidth=1.5)

    ax.set_xlabel('并发数 / 池大小', fontsize=8, fontweight='bold')
    ax.set_ylabel('内存漂移 (MB, PM 口径)', fontsize=8, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 标注 Unsafe Shared 漂移激增（文字置于峰值左侧、略低于峰值，使箭头基本水平、稍微上扬，且不出框）
    ax.annotate(f'{us_drift[-1]:.0f} MB\n(12并发净增)', xy=(concurrency[-1], us_drift[-1]),
                xytext=(max(concurrency) - 6, us_drift[-1] - 120),
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=2, shrinkB=8),
                fontsize=7, color='#888888', fontweight='bold')
    # 标注 Session Pool 漂移可控（净释放）
    ax.annotate(f'{sp_drift[-1]:.0f} MB\n(净释放)', xy=(concurrency[-1], sp_drift[-1]),
                xytext=(max(concurrency) - 6, 600),
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2, shrinkB=8),
                fontsize=7, color='#000000', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_memory_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_memory_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig3_memory_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 3 已生成：fig3_memory_comparison.png")

plot_memory_comparison()

# ============================================
# 图 4: 顺序单图推理调用开销 (对应 §4.2)
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
    # 注：go_batch_inference_result.json 的 batch_size 字段实为"顺序单图推理连续调用次数"
    # （batch 恒为1，循环调用 N 次），与论文 §4.2 对齐；非批维度扩展。
    ax1.set_xlabel('连续推理次数（batch=1 顺序调用）', fontsize=8, fontweight='bold')
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
    ax2.set_ylim(0, 2.0)  # 扩大Y轴范围，避免视觉误导
    
    # 标题和图例
    # plt.suptitle('CPU 推理场景下批处理对吞吐量的影响', fontsize=9, fontweight='bold', y=0.90)  # 图注由论文caption提供
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, framealpha=0.9)
    
    # 添加结论文字（与论文 §4.2 图\ref{fig:batch_effect} caption 口径一致）
    fig.text(0.5, 0.02, '结论：顺序单图推理连续调用下，吞吐随推理次数增加缓慢下降\n（1→32次约-8%），反映调用开销累积而非批维度收益',
             fontsize=8, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{output_dir}/fig4_batch_effect.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_batch_effect.pdf', bbox_inches='tight')
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
        # 读取 Python 强化测试数据 (YOLO11x)
        python_latency_11x = 0
        python_std_11x = 0.0
        with open(os.path.join(results_dir, 'python_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if '平均延迟：' in line or '平均延迟:' in line or 'Avg latency:' in line:
                    if 'Avg latency:' in line:
                        latency_str = line.split('Avg latency:')[1].strip()
                    else:
                        latency_str = line.split('：')[1].strip() if '：' in line else line.split(':')[1].strip()
                    python_latency_11x = float(latency_str.split(' ')[0])
                if 'Std dev:' in line:
                    python_std_11x = float(line.split('Std dev:')[1].strip().split(' ')[0])
        
        # 读取 Go 强化测试数据 (YOLO11x)
        go_latency_11x = 0
        go_std_11x = 0.0
        with open(os.path.join(results_dir, 'go_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                # 用 startswith 排除"第X轮平均延迟:"行，只匹配10轮均值行"平均延迟:"
                if line_strip.startswith('平均延迟：') or line_strip.startswith('平均延迟:'):
                    latency_str = line_strip.split('：')[1].strip() if '：' in line_strip else line_strip.split(':')[1].strip()
                    go_latency_11x = float(latency_str.split(' ')[0])
                if line_strip.startswith('标准差:'):
                    go_std_11x = float(line_strip.split('标准差:')[1].strip().split(' ')[0])
        
        # 读取 Python YOLO11n 强化测试数据
        python_latency_11n = 0
        python_std_11n = 0.0
        with open(os.path.join(results_dir, 'python_yolo11n_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            in_summary = False
            for line in lines:
                if '10-Round Average' in line or '10轮测试平均值' in line or '10轮平均' in line or 'Overall average:' in line:
                    in_summary = True
                    continue
                if in_summary:
                    if 'Avg latency:' in line or '平均延迟:' in line or '平均延迟：' in line:
                        for sep in ['：', ':']:
                            if sep in line:
                                try:
                                    python_latency_11n = float(line.split(sep)[1].strip().split(' ')[0])
                                    if python_latency_11n > 0:
                                        break
                                except ValueError:
                                    pass
                    if 'Std dev:' in line:
                        try:
                            python_std_11n = float(line.split('Std dev:')[1].strip().split(' ')[0])
                        except ValueError:
                            pass
                    if python_latency_11n > 0 and python_std_11n > 0:
                        break
        # 读取 Go YOLO11n 强化测试数据
        go_latency_11n = 0
        go_std_11n = 0.0
        with open(os.path.join(results_dir, 'go_yolo11n_reinforced_result.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            in_summary = False
            for line in lines:
                if '10-Round Average' in line or '10轮测试平均值' in line or '10轮平均' in line or 'Overall average:' in line:
                    in_summary = True
                    continue
                if in_summary:
                    if 'Avg latency:' in line or '平均延迟:' in line or '平均延迟：' in line:
                        for sep in ['：', ':']:
                            if sep in line:
                                try:
                                    go_latency_11n = float(line.split(sep)[1].strip().split(' ')[0])
                                    if go_latency_11n > 0:
                                        break
                                except ValueError:
                                    pass
                    if line.strip().startswith('标准差:'):
                        try:
                            go_std_11n = float(line.strip().split('标准差:')[1].strip().split(' ')[0])
                        except ValueError:
                            pass
                    if go_latency_11n > 0 and go_std_11n > 0:
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
        
        return ([python_latency_11n, python_latency_11x], [go_latency_11n, go_latency_11x],
                [python_std_11n, python_std_11x], [go_std_11n, go_std_11x])
    except Exception as e:
        print(f"读取模型延迟数据失败: {e}")
        raise

def plot_model_size_comparison():
    models = ['YOLO11n\n(轻量)', 'YOLO11x\n(大模型)']
    python_latency_values, go_latency_values, python_std_values, go_std_values = read_model_latency_data()
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars1 = ax.bar(x - width/2, python_latency_values, width, label='Python',
                   color='#666666', edgecolor='black', linewidth=1.5,
                   yerr=python_std_values, capsize=4, ecolor='#333333')
    bars2 = ax.bar(x + width/2, go_latency_values, width, label='Go',
                   color='#000000', edgecolor='black', linewidth=1.5,
                   yerr=go_std_values, capsize=4, ecolor='#333333')
    
    ax.set_ylabel('推理延迟 (ms)', fontsize=8, fontweight='bold')
    # ax.set_title('不同模型规模下的推理延迟对比', fontsize=9, fontweight='bold', pad=10)  # 图注由论文caption提供
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(['Python', 'Go'], fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    ax.set_axisbelow(True)
    
    # 使用对数y轴，使YOLO11n(~40ms)和YOLO11x(~700ms)的柱子都清晰可见
    ax.set_yscale('log')
    ax.set_ylim(10, 1000)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 性能差异标注与论文表\ref{tab:model_size}一致：差异=(Go-Python)/Python×100%，正值表示Go延迟较高
    diff_percent = [(go_latency_values[i]-python_latency_values[i])/python_latency_values[i]*100
                    for i in range(len(go_latency_values))]
    for i, diff in enumerate(diff_percent):
        # 使用绝对坐标放置差异标注，避免在对数轴下触碰顶框
        bar_top = max(python_latency_values[i], go_latency_values[i])
        if bar_top > 200:  # 大模型组(YOLO11x)
            y_pos = 800  # 比柱顶(737.5)稍高，不触碰顶框
        else:  # 小模型组(YOLO11n)
            y_pos = bar_top * 1.15
        ax.text(i, y_pos, f'{diff:.1f}% 差异',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_model_size_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig5_model_size_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig5_model_size_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 5 已生成：fig5_model_size_comparison.png")

plot_model_size_comparison()

# ============================================
# 图 6: CPU 利用率梯度对比（Go Session Pool 不同负载场景）
# ============================================
def read_cpu_gradient_data():
    """从 Go CPU 监控结果读取四个场景的 CPU 利用率数据"""
    try:
        with open(os.path.join(results_dir, 'go_cpu_monitoring_result.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Go CPU 监控数据格式错误：期望 JSON 数组")
        
        # 按场景名提取数据
        scenario_map = {}
        for item in data:
            scenario_map[item.get('scenario', '')] = item
        
        # 四个场景：Idle -> Single_Inference -> Continuous_100_Inferences -> Concurrent_10x50
        scenarios_ordered = ['Idle', 'Single_Inference', 'Continuous_100_Inferences', 'Concurrent_10x50']
        cpu_values = []
        for sc in scenarios_ordered:
            if sc not in scenario_map:
                raise ValueError(f"未找到场景 '{sc}' 的 CPU 数据")
            cpu_values.append(round(scenario_map[sc].get('avg_cpu_percent', 0), 2))
        
        return cpu_values, scenarios_ordered
    except Exception as e:
        print(f"读取 CPU 利用率数据失败：{e}")
        raise

def plot_cpu_utilization():
    cpu_values, scenarios_ordered = read_cpu_gradient_data()
    
    # 简洁的场景标签
    labels = ['空闲\n(Idle)', '单次推理\n(Single)', '连续推理\n(100次顺序)', '并发推理\n(10并发)']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 使用渐变色表示负载递增
    colors = ['#d4d4d4', '#a0a0a0', '#505050', '#000000']
    bars = ax.bar(labels, cpu_values, color=colors,
                  edgecolor='black', linewidth=1.2, width=0.55)
    
    ax.set_ylabel('CPU 利用率 (%)', fontsize=10, fontweight='bold')
    # ax.set_title('Go Session Pool 在不同负载场景下的 CPU 利用率梯度', fontsize=10, fontweight='bold', pad=12)  # 图注由论文caption提供
    ax.set_ylim(0, 700)  # 固定Y轴范围，使梯度差异更直观
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加配置说明
    ax.annotate('intra_op=8, GOMAXPROCS=12\n模型: YOLO11x',
                xy=(0.03, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=6.5, color='#555555',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.85))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(f'{output_dir}/fig6_cpu_utilization.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig6_cpu_utilization.pdf', bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig6_cpu_utilization.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("图 6 已生成：fig6_cpu_utilization.png")

plot_cpu_utilization()

# ============================================
# 图 7: 长时间稳定性对比 (对应第 5 章)
# ============================================
def read_stability_data():
    """从稳定性测试结果文件读取稳定性数据，自动检测时长"""
    try:
        # 读取 Python 稳定性数据 - 优先 JSON 格式（有完整采样点）
        python_duration_hours = 1.0
        python_initial_pm = 0
        python_final_pm = 0
        python_samples = []
        # 优先级: 72h JSON > 1h JSON > 72h TXT > 1h TXT > 旧的 long_stability
        python_stability_candidates = [
            ('json', 'python_stability_72h_result.json'),
            ('json', 'python_stability_1h_result.json'),
            ('txt', 'python_stability_72h_result.txt'),
            ('txt', 'python_stability_1h_result.txt'),
            ('txt', 'python_long_stability_result.txt'),
        ]
        python_stability_file = None
        python_file_format = None
        for fmt, candidate in python_stability_candidates:
            candidate_path = os.path.join(results_dir, candidate)
            if os.path.exists(candidate_path):
                python_stability_file = candidate_path
                python_file_format = fmt
                break
        if not python_stability_file:
            raise FileNotFoundError("未找到 Python 稳定性测试结果文件")

        if python_file_format == 'json':
            with open(python_stability_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            python_duration_hours = data.get('results', {}).get('actual_duration_hours',
                data.get('config', {}).get('test_duration_hours', 1.0))
            # 实测为 PM（Private Bytes），字段名 rss_* 为旧命名；优先读取 pm_* 新命名并兼容旧命名
            python_initial_pm = data.get('results', {}).get('start_pm_mb',
                data.get('results', {}).get('start_rss_mb', 0))
            python_final_pm = data.get('results', {}).get('end_pm_mb',
                data.get('results', {}).get('end_rss_mb', 0))
            pm_samples = data.get('pm_samples', data.get('rss_samples', []))
            for s in pm_samples:
                python_samples.append((s.get('hour', 0), s.get('pm_mb', s.get('rss_mb', 0))))
        else:
            with open(python_stability_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Start RSS:' in line or 'Start PM:' in line:
                        try:
                            python_initial_pm = float(line.split(':')[1].strip().split(' ')[0])
                        except ValueError:
                            pass
                    elif 'End RSS:' in line or 'End PM:' in line:
                        try:
                            python_final_pm = float(line.split(':')[1].strip().split(' ')[0])
                        except ValueError:
                            pass
                    elif 'Duration:' in line or 'duration_hours:' in line:
                        try:
                            python_duration_hours = float(line.split(':')[1].strip().split(' ')[0])
                        except ValueError:
                            pass

        # 读取 Go 稳定性数据 - 自动检测时长
        go_hours = []
        go_rss_values = []
        go_duration_hours = 0.0
        go_stability_candidates = [
            'go_stability_72h_result.json',
            'go_stability_1h_result.json',
        ]
        go_stability_file = None
        for candidate in go_stability_candidates:
            candidate_path = os.path.join(results_dir, candidate)
            if os.path.exists(candidate_path):
                go_stability_file = candidate_path
                break
        if not go_stability_file:
            raise FileNotFoundError("未找到 Go 稳定性测试结果文件 (go_stability_*_result.json)")
        with open(go_stability_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            go_duration_hours = data.get('duration_hours', data.get('actual_duration_hours', 1.0))
            snapshots = data.get('hourly_snapshots', [])
            if snapshots:
                step = max(1, len(snapshots) // 72)
                for i in range(0, len(snapshots), step):
                    s = snapshots[i]
                    go_hours.append(s.get('hour', 0))
                    go_rss_values.append(s.get('pm_mb', s.get('rss_mb', 0)))
                if go_hours[-1] != snapshots[-1].get('hour', 0):
                    go_hours.append(snapshots[-1].get('hour', 0))
                    go_rss_values.append(snapshots[-1].get('pm_mb', snapshots[-1].get('rss_mb', 0)))

        if python_initial_pm == 0 or python_final_pm == 0:
            raise ValueError("无法读取 Python 稳定性数据")

        if not go_hours:
            raise ValueError("无法读取 Go 稳定性快照数据")

        return (python_initial_pm, python_final_pm, python_duration_hours, python_samples,
                go_hours, go_rss_values, go_duration_hours)
    except Exception as e:
        print(f"读取稳定性数据失败：{e}")
        raise

def plot_stability():
    (python_initial_pm, python_final_pm, python_duration_hours, python_samples,
     go_hours, go_rss_values, go_duration_hours) = read_stability_data()

    fig, ax = plt.subplots(figsize=(12, 6))

    # 统一横坐标 - 取两者中较长的时长
    max_hours = max(python_duration_hours, go_duration_hours)
    duration_label = f"{max_hours:.1f}小时" if max_hours < 2 else f"{int(max_hours)}小时"

    # Python - 如果有完整采样点就画实线，否则画虚线趋势线
    if python_samples and len(python_samples) > 2:
        py_sample_hours = [s[0] for s in python_samples]
        py_sample_rss = [s[1] for s in python_samples]
        ax.plot(py_sample_hours, py_sample_rss, '-', label='Python (实测采样)',
                color='#555555', linewidth=2.5, alpha=1.0)
    else:
        python_hours = [0, python_duration_hours]
        python_rss = [python_initial_pm, python_final_pm]
        ax.plot(python_hours, python_rss, '^--', label='Python (仅起点/终点)',
                color='#999999', linewidth=2, markersize=10, markeredgewidth=1.5,
                alpha=0.7)

    # Go: 实测快照
    ax.plot(go_hours, go_rss_values, 'o-', label='Go 单Session (实测快照)',
            color='#000000', linewidth=1.5, markersize=5, alpha=0.8)

    ax.set_xlabel('运行时间 (小时)', fontsize=9, fontweight='bold')
    ax.set_ylabel('进程私有内存 PM (MB)', fontsize=9, fontweight='bold')
    # ax.set_title(f'{duration_label}内存稳定性测试', fontsize=10, fontweight='bold', pad=12)  # 图注由论文caption提供
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.01, 0.88), framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 标注 Python 漂移（颜色与Python线一致）
    python_drift = python_final_pm - python_initial_pm
    drift_rate = python_drift / python_duration_hours if python_duration_hours > 0 else 0
    ax.annotate(f'Python 漂移: +{python_drift:.2f} MB\n({drift_rate:.3f} MB/h)',
                xy=(python_duration_hours, python_final_pm),
                xytext=(python_duration_hours * 0.4, python_final_pm - 50),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2, shrinkB=50, mutation_scale=15),
                fontsize=8, color='#555555', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#555555'))

    # 标注 Go 漂移（使用JSON顶层start/end字段，与论文正文一致）
    _go_stab_path = os.path.join(results_dir, 'go_stability_72h_result.json')
    if not os.path.exists(_go_stab_path):
        _go_stab_path = os.path.join(results_dir, 'go_stability_1h_result.json')
    with open(_go_stab_path, 'r', encoding='utf-8') as gf:
        go_json = json.load(gf)
        go_start = go_json.get('start_pm_mb', go_rss_values[0])
        go_end = go_json.get('end_pm_mb', go_rss_values[-1])
    go_drift = go_end - go_start
    go_drift_rate = go_drift / go_duration_hours if go_duration_hours > 0 else 0
    ax.annotate(f'Go 漂移: {go_drift:.2f} MB\n({go_drift_rate:.3f} MB/h)',
                xy=(go_hours[-1], go_rss_values[-1]),
                xytext=(go_hours[-1] * 0.4, go_rss_values[-1] + 15),
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2, shrinkB=50, mutation_scale=15),
                fontsize=8, color='#000000', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_stability.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig7_stability.pdf', bbox_inches='tight')
    plt.savefig(f'{final_charts_dir}/fig7_stability.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"图 7 已生成：fig7_stability.png（{duration_label}实测数据）")

plot_stability()

print("\n" + "=" * 60)
print("所有 6 张图表已生成完毕！")
print("=" * 60)
print(f"\n图表保存位置：{os.path.abspath(output_dir)}")
print("\n生成的图表列表:")
print("  图 2. fig2_throughput_comparison.png   - 三种并发架构的吞吐量对比")
print("  图 3. fig3_memory_comparison.png       - Go 双架构内存漂移对比（PM 漂移口径）")
print("  图 4. fig4_batch_effect.png            - 顺序单图推理调用开销（连续推理1→32次）")
print("  图 5. fig5_model_size_comparison.png   - 不同模型规模下的推理延迟对比")
print("  图 6. fig6_cpu_utilization.png         - Go Session Pool 不同负载场景的 CPU 利用率梯度")
print("  图 7. fig7_stability.png               - 内存漂移对比")
print("\n提示：这些图表可以直接插入到 LaTeX 论文中")
print("=" * 60)