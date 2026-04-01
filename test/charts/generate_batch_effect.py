import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从JSON文件中读取批处理数据
try:
    with open('../../results/go_batch_inference_result.json', 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    batch_size = []
    latency = []
    throughput = []
    
    if 'results' not in batch_data:
        raise ValueError("JSON 文件中缺少'results'字段")
    
    for item in batch_data['results']:
        batch_size.append(item['batch_size'])
        latency.append(item['per_image_time_ms'])
        throughput.append(item['throughput_images_per_sec'])
    
    if not batch_size:
        raise ValueError("批处理数据为空")
        
except FileNotFoundError:
    raise FileNotFoundError("无法读取批处理结果文件：go_batch_inference_result.json")
except json.JSONDecodeError as e:
    raise ValueError(f"JSON 文件格式错误：{e}")
except KeyError as e:
    raise KeyError(f"JSON 文件中缺少必需字段：{e}")
except Exception as e:
    raise RuntimeError(f"读取批处理数据失败：{e}")

# 绘图
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'k'  # 黑色
color2 = 'gray'

# 延迟曲线
ax1.plot(batch_size, latency, 'k--o', label='单图延迟 (ms)')
ax1.set_xlabel('批处理大小', fontsize=11)
ax1.set_ylabel('单图延迟 (ms)', fontsize=11, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 吞吐量曲线（共享 X 轴）
ax2 = ax1.twinx()
ax2.plot(batch_size, throughput, 'k-.s', label='吞吐量 (img/s)', linewidth=2)
ax2.set_ylabel('吞吐量 (img/s)', fontsize=11, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 标题
ax1.set_title('CPU 推理场景下批处理对吞吐量的影响', fontsize=12)

# 图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

ax1.grid(True, linestyle=':', color='gray', alpha=0.6)

# 保存图片
plt.savefig("batch_effect_cpu.png", dpi=600, bbox_inches='tight', format='png')
plt.show()