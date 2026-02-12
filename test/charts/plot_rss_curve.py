import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为华文中宋，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['font.family'] = ['sans-serif', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 读取你已经生成的 CSV
go_rss = pd.read_csv("../../results/go_rss_curve.csv")
py_rss = pd.read_csv("../../results/python_rss_curve.csv")

plt.figure(figsize=(7, 4.5))

plt.plot(go_rss["Elapsed_Seconds"], go_rss["RSS_MB"], label="Go")
plt.plot(py_rss["Elapsed_Seconds"], py_rss["RSS_MB"], label="Python")

plt.xlabel("Time (s)")
plt.ylabel("RSS Memory (MB)")
plt.title("RSS Memory Usage During Long-Term Inference")
plt.legend()
plt.grid(linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("../../results/rss_curve.pdf")
plt.savefig("../../results/charts/rss_curve.png")
print("内存使用曲线已生成: ../../results/rss_curve.pdf")
print("内存使用曲线(PNG)已生成: ../../results/charts/rss_curve.png")
plt.show()
