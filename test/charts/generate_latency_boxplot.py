import matplotlib.pyplot as plt
import re

# 设置中文字体为华文中宋，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['font.family'] = ['sans-serif', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Go 基准测试结果
with open("../../results/go_baseline_latency_data.txt", "r", encoding="utf-8") as f:
    go_latency = [float(line.strip()) for line in f if line.strip()]

# 读取 Python 基准测试结果
with open("../../results/python_baseline_latency_data.txt", "r", encoding="utf-8") as f:
    py_latency = [float(line.strip()) for line in f if line.strip()]

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

# 设置箱线图颜色
colors = ['#66c2a5', '#fc8d62']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel("Inference Latency (ms)", fontsize=12)
plt.title("Inference Latency Distribution Comparison", fontsize=14, fontweight='bold')
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("../../results/latency_boxplot.pdf", dpi=300, bbox_inches='tight')
plt.savefig("../../results/charts/latency_boxplot.png", dpi=300, bbox_inches='tight')
print("延迟箱线图已生成: ../../results/latency_boxplot.pdf")
print("延迟箱线图(PNG)已生成: ../../results/charts/latency_boxplot.png")
plt.show()
