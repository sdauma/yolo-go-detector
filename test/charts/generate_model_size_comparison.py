import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从文件中读取延迟数据
def read_latency(file_path):
    """从文件中读取延迟数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '平均延迟:' in line or '平均延迟：' in line or 'Avg latency:' in line:
                    for sep in [':', '：']:
                        if sep in line:
                            latency_str = line.split(sep)[1].strip().split(' ')[0]
                            try:
                                latency = float(latency_str)
                                return latency
                            except ValueError:
                                pass
        raise ValueError(f"未在文件 {file_path} 中找到延迟数据")
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取延迟数据失败：{e}")

# 读取数据
try:
    # 读取Python YOLO11n延迟
    python_yolo11n = read_latency('../../results/python_yolo11n_reinforced_result.txt')
    
    # 读取Python YOLO11x延迟
    python_yolo11x = read_latency('../../results/python_reinforced_result.txt')
    
    # 读取Go YOLO11n延迟
    go_yolo11n = read_latency('../../results/go_reinforced_small_result.txt')
    
    # 读取Go YOLO11x延迟
    go_yolo11x = read_latency('../../results/go_reinforced_result.txt')
except Exception as e:
    print(f"错误：{e}")
    raise

# 模型名称
models = ['YOLO11n', 'YOLO11x']

# 对应延迟 (ms)
python_latency = [python_yolo11n, python_yolo11x]
go_latency = [go_yolo11n, go_yolo11x]

x = np.arange(len(models))  # x轴位置
width = 0.35  # 柱宽

fig, ax = plt.subplots(figsize=(7, 5))

# 绘制柱状图
rects1 = ax.bar(x - width/2, python_latency, width, label='Python', color='white', edgecolor='k', hatch='/')
rects2 = ax.bar(x + width/2, go_latency, width, label='Go', color='gray', edgecolor='k', hatch='//')

# 添加标签和标题
ax.set_ylabel('推理延迟 (ms)', fontsize=11)
ax.set_xlabel('模型类型', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_title('不同模型规模下的推理延迟对比', fontsize=12)
ax.legend(['Python', 'Go'], fontsize=10, prop={'family': 'Times New Roman'})
ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.6)

# 在柱上显示数值
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}', 
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# 保存图片
plt.savefig("model_size_latency.png", dpi=600, bbox_inches='tight', format='png')