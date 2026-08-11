import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import sys

results_dir = '../../results'

# 与论文主图生成脚本(generate_all_charts.py)保持一致：用 font_utils 显式注册宋体
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import font_utils
font_utils.setup_fonts()

def parse_arena_result(filepath):
    results = []
    current = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=====' in line and 'Unsafe Shared' in line:
                current = {'arch': 'Unsafe Shared', 'arena': 'ON' if 'arena=ON' in line else 'OFF'}
            elif '=====' in line and 'Session Pool' in line:
                current = {'arch': 'Session Pool', 'arena': 'ON' if 'arena=ON' in line else 'OFF'}
            elif current and '吞吐量:' in line:
                current['throughput'] = float(line.split(':')[1].strip().split(' ')[0])
            elif current and '平均延迟:' in line:
                current['avg_latency'] = float(line.split(':')[1].strip().split(' ')[0])
            elif current and ('峰值RSS:' in line or '峰值PM:' in line):
                current['peak_rss'] = float(line.split(':')[1].strip().split(' ')[0])
            elif current and ('RSS漂移:' in line or 'PM漂移:' in line):
                current['rss_drift'] = float(line.split(':')[1].strip().split(' ')[0])
                results.append(current)
                current = None
    return results

go_data = parse_arena_result(os.path.join(results_dir, 'go_arena_ablation_result.txt'))
py_data = parse_arena_result(os.path.join(results_dir, 'python_arena_ablation_result.txt'))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

archs = ['Unsafe Shared', 'Session Pool']
arena_labels = ['Arena ON', 'Arena OFF']
x = range(len(archs))
width = 0.3

for i, metric in enumerate(['throughput', 'peak_rss', 'rss_drift']):
    ax = axes[i]
    
    go_on = [d[metric] for d in go_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'ON'][0]
    go_off = [d[metric] for d in go_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'OFF'][0]
    sp_on = [d[metric] for d in go_data if d['arch'] == 'Session Pool' and d['arena'] == 'ON'][0]
    sp_off = [d[metric] for d in go_data if d['arch'] == 'Session Pool' and d['arena'] == 'OFF'][0]
    
    go_vals_on = [go_on, sp_on]
    go_vals_off = [go_off, sp_off]
    
    bars1 = ax.bar([xi - width/2 for xi in x], go_vals_on, width, label='Arena ON', color='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='//')
    bars2 = ax.bar([xi + width/2 for xi in x], go_vals_off, width, label='Arena OFF', color='#B8B8B8', edgecolor='black', linewidth=0.5, hatch='\\\\')
    
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if metric == 'throughput':
        ax.set_ylabel('吞吐量 (REQ/s)', fontsize=10)
        ax.set_title('(a) 吞吐量对比', fontsize=11)
    elif metric == 'peak_rss':
        ax.set_ylabel('峰值 PM (MB)', fontsize=10)
        ax.set_title('(b) 峰值内存对比', fontsize=11)
    elif metric == 'rss_drift':
        ax.set_ylabel('PM 漂移 (MB)', fontsize=10)
        ax.set_title('(c) 内存漂移对比', fontsize=11)
    
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('图 4-补充  Arena 开关消融实验对比（Go 侧，4并发/4池）', fontsize=12, y=1.02)
plt.tight_layout()

out_png = os.path.join(results_dir, 'charts', 'arena_ablation_comparison.png')
out_pdf = os.path.join(results_dir, 'charts', 'arena_ablation_comparison.pdf')
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Arena 消融实验对比图已保存: {out_png}')
print(f'Arena 消融实验对比图已保存: {out_pdf}')

fig2, ax2 = plt.subplots(figsize=(8, 5))

py_us_on = [d['rss_drift'] for d in py_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'ON'][0]
py_us_off = [d['rss_drift'] for d in py_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'OFF'][0]
go_us_on = [d['rss_drift'] for d in go_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'ON'][0]
go_us_off = [d['rss_drift'] for d in go_data if d['arch'] == 'Unsafe Shared' and d['arena'] == 'OFF'][0]

groups = ['Go (变量可控)', 'Python (变量不可控)']
on_vals = [go_us_on, py_us_on]
off_vals = [go_us_off, py_us_off]

x2 = range(len(groups))
bars3 = ax2.bar([xi - width/2 for xi in x2], on_vals, width, label='Arena ON', color='#E8E8E8', edgecolor='black', linewidth=0.5, hatch='//')
bars4 = ax2.bar([xi + width/2 for xi in x2], off_vals, width, label='Arena OFF', color='#B8B8B8', edgecolor='black', linewidth=0.5, hatch='\\\\')

ax2.set_xticks(x2)
ax2.set_xticklabels(groups, fontsize=11)
ax2.set_ylabel('PM 漂移 (MB)', fontsize=11)
ax2.set_title('图 4-补充(b)  跨语言负对照验证：Arena 开关对 PM 漂移的影响', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}', ha='center', va='bottom', fontsize=9)

ax2.annotate('', xy=(0, go_us_off), xytext=(0, go_us_on),
             arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax2.text(0.05, (go_us_on + go_us_off)/2, f'下降{(1-go_us_off/go_us_on)*100:.1f}%', color='black', fontsize=9, va='center')

ax2.annotate('', xy=(1, py_us_off), xytext=(1, py_us_on),
             arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, linestyle='--'))
ax2.text(1.05, (py_us_on + py_us_off)/2, f'差异{(abs(py_us_on-py_us_off)/py_us_on)*100:.1f}%', color='gray', fontsize=9, va='center')

plt.tight_layout()
out_png2 = os.path.join(results_dir, 'charts', 'arena_cross_language_validation.png')
out_pdf2 = os.path.join(results_dir, 'charts', 'arena_cross_language_validation.pdf')
plt.savefig(out_png2, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf2, bbox_inches='tight')
print(f'跨语言负对照验证图已保存: {out_png2}')
print(f'跨语言负对照验证图已保存: {out_pdf2}')

print('\n=== 数据汇总 ===')
print('\nGo 侧:')
for d in go_data:
    print(f"  {d['arch']} Arena={d['arena']}: 吞吐={d['throughput']:.3f}, 峰值PM={d['peak_rss']:.1f}MB, 漂移={d['rss_drift']:.2f}MB")

print('\nPython 侧:')
for d in py_data:
    print(f"  {d['arch']} Arena={d['arena']}: 吞吐={d['throughput']:.3f}, 峰值PM={d['peak_rss']:.1f}MB, 漂移={d['rss_drift']:.2f}MB")
