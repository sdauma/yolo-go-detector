import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从原始数据文件读取内存数据
def read_memory_data(file_path):
    """从内存测试结果文件中读取平均内存值"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找大模型(YOLO11x)的平均值
    inference_memory = None
    for i, line in enumerate(lines):
        if '大模型 (YOLO11x) 10次测试平均值' in line:
            # 查找推理后内存
            for j in range(i+1, min(i+10, len(lines))):
                if '推理后内存:' in lines[j]:
                    # 提取内存值
                    parts = lines[j].split(':')
                    if len(parts) > 1:
                        inference_memory = float(parts[1].strip().split(' ')[0])
                        break
            break
    
    return inference_memory

# 读取内存数据
import os
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
go_memory_file = os.path.join(project_root, 'results', 'go_memory_standardization_result.txt')
python_memory_file = os.path.join(project_root, 'results', 'python_memory_standardization_result.txt')

go_inference_memory = read_memory_data(go_memory_file)
python_inference_memory = read_memory_data(python_memory_file)

# 配置参数
server_memory = 128 * 1024  # 服务器内存 (MB)
python_instances = 374  # Python所需实例数
go_instances = 486  # Go所需实例数
python_servers = 2  # Python所需服务器数
go_servers = 1  # Go所需服务器数

# 计算总内存占用
python_total_memory = python_instances * python_inference_memory
go_total_memory = go_instances * go_inference_memory

# 计算内存利用率
python_utilization = (python_total_memory / (python_servers * server_memory)) * 100
go_utilization = (go_total_memory / (go_servers * server_memory)) * 100

# 数据配置
python_used = python_utilization  # Python已使用内存
python_free = 100 - python_used  # Python冗余内存
go_used = go_utilization  # Go已使用内存
go_free = 100 - go_used  # Go冗余内存

# 颜色配置
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Python方案饼图
ax1.pie([python_used, python_free], labels=['已使用', '冗余'], autopct='%1.1f%%', 
        startangle=90, colors=colors[:2], wedgeprops={'edgecolor': 'white'})
ax1.set_title('Python方案内存利用率', fontsize=14)
ax1.axis('equal')  # 确保饼图是圆的

# Go方案饼图
ax2.pie([go_used, go_free], labels=['已使用', '冗余'], autopct='%1.1f%%', 
        startangle=90, colors=colors[2:], wedgeprops={'edgecolor': 'white'})
ax2.set_title('Go方案内存利用率', fontsize=14)
ax2.axis('equal')  # 确保饼图是圆的

# 整体标题
plt.suptitle('两种方案的内存利用率对比', fontsize=16)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图表
output_path = os.path.join(project_root, 'results', 'charts', 'fig8_memory_utilization_pie.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight')

# 打印计算结果
print(f"=== 内存利用率计算结果 ===")
print(f"Go单实例内存: {go_inference_memory:.2f} MB")
print(f"Python单实例内存: {python_inference_memory:.2f} MB")
print(f"Go总内存占用: {go_total_memory:.2f} MB ({go_total_memory/1024:.2f} GB)")
print(f"Python总内存占用: {python_total_memory:.2f} MB ({python_total_memory/1024:.2f} GB)")
print(f"Go内存利用率: {go_utilization:.2f}%")
print(f"Python内存利用率: {python_utilization:.2f}%")
print(f"图表已保存到: {output_path}")

# 显示图表（可选）
# plt.show()