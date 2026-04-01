import matplotlib.pyplot as plt
import re
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 读取 Go 基准测试结果
try:
    with open("../../results/go_baseline_latency_data.txt", "r", encoding="utf-8") as f:
        go_latency = [float(line.strip()) for line in f if line.strip()]
    
    if not go_latency:
        raise ValueError("Go 延迟数据为空")
except FileNotFoundError:
    raise FileNotFoundError("无法读取 Go 延迟数据文件：go_baseline_latency_data.txt")
except Exception as e:
    raise RuntimeError(f"读取 Go 延迟数据失败：{e}")

# 读取 Python 基准测试结果
try:
    with open("../../results/python_baseline_latency_data.txt", "r", encoding="utf-8") as f:
        py_latency = [float(line.strip()) for line in f if line.strip()]
    
    if not py_latency:
        raise ValueError("Python 延迟数据为空")
except FileNotFoundError:
    raise FileNotFoundError("无法读取 Python 延迟数据文件：python_baseline_latency_data.txt")
except Exception as e:
    raise RuntimeError(f"读取 Python 延迟数据失败：{e}")

print(f"Go 延迟数据: {len(go_latency)} 次")
print(f"Python 延迟数据: {len(py_latency)} 次")
print(f"Go 平均延迟: {sum(go_latency)/len(go_latency):.3f} ms")
print(f"Python 平均延迟: {sum(py_latency)/len(py_latency):.3f} ms")

# 创建箱线图
plt.figure(figsize=(8, 5))
box = plt.boxplot(
    [go_latency, py_latency],
    tick_labels=["Go + ONNX Runtime", "Python + ONNX Runtime"],
    showfliers=True,
    patch_artist=True,
    widths=0.6
)

# 设置箱线图颜色（黑白印刷友好）
for patch, color in zip(box['boxes'], ['#E0E0E0', '#B0B0B0']):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)
    patch.set_hatch('//')

plt.ylabel("推理延迟 (ms)", fontsize=12)
plt.title("推理延迟分布对比", fontsize=14, fontweight='bold')
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("../../results/latency_boxplot.pdf", dpi=600, bbox_inches='tight')
plt.savefig("../../results/charts/latency_boxplot.png", dpi=600, bbox_inches='tight')
print("延迟箱线图已生成: ../../results/latency_boxplot.pdf")
print("延迟箱线图(PNG)已生成: ../../results/charts/latency_boxplot.png")
plt.show()
