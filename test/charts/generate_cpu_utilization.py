import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从文件中读取CPU利用率数据
def read_cpu_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Average CPU:' in line or '平均 CPU' in line:
                    cpu_util = float(line.split(':')[1].strip().replace('%', ''))
                    return cpu_util
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取 CPU 数据文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取 CPU 数据失败：{e}")
    
    raise FileNotFoundError(f"未在文件 {file_path} 中找到 CPU 利用率数据")

# 读取Python CPU数据
python_cpu = read_cpu_data('../../results/python_cpu_monitoring_result.txt')

# 读取Go CPU数据
go_cpu = read_cpu_data('../../results/go_cpu_monitoring_result.txt')

# 架构名称
archs = ['Python 单实例', 'Go Session 池']

# CPU 利用率 (%)
cpu_util = [python_cpu, go_cpu]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(archs, cpu_util, color='white', edgecolor='k', hatch='//')

# 标签和标题
ax.set_ylabel('CPU 利用率 (%)', fontsize=11)
ax.set_title('不同部署方案的 CPU 利用率对比', fontsize=12)
ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.6)

# 在柱上显示数值
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', 
                xy=(bar.get_x() + bar.get_width()/2, height), 
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# 保存图片
plt.savefig("cpu_utilization.png", dpi=600, bbox_inches='tight', format='png')
plt.show()