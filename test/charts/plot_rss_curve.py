import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为SimHei以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取已经生成的 CSV
try:
    go_rss = pd.read_csv("../../results/go_rss_curve.csv")
    if go_rss.empty:
        raise ValueError("Go 内存数据为空")
except FileNotFoundError:
    raise FileNotFoundError("无法读取 Go 内存数据文件：go_rss_curve.csv")
except Exception as e:
    raise RuntimeError(f"读取 Go 内存数据失败：{e}")

try:
    py_rss = pd.read_csv("../../results/python_rss_curve.csv")
    if py_rss.empty:
        raise ValueError("Python 内存数据为空")
except FileNotFoundError:
    raise FileNotFoundError("无法读取 Python 内存数据文件：python_rss_curve.csv")
except Exception as e:
    raise RuntimeError(f"读取 Python 内存数据失败：{e}")

plt.figure(figsize=(7, 4.5))

go_time_min = go_rss["Elapsed_Seconds"] / 60.0
py_time_min = py_rss["Elapsed_Seconds"] / 60.0

plt.plot(go_time_min, go_rss["RSS_MB"], 'k--o', label="Go", linewidth=1.5, markersize=4)
plt.plot(py_time_min, py_rss["RSS_MB"], 'k-.s', label="Python", linewidth=1.5, markersize=4)

plt.xlabel("运行时间 (min)", fontsize=11)
plt.ylabel("内存占用 (MB)", fontsize=11)
plt.title("长时间运行的内存漂移对比", fontsize=12)
plt.legend(fontsize=10)
plt.grid(linestyle=":", linewidth=0.5, color='gray', alpha=0.6)

plt.tight_layout()
plt.savefig("../../results/rss_curve.pdf", dpi=600)
plt.savefig("../../results/charts/fig7_stability.png", dpi=600)
print("内存使用曲线已生成: ../../results/rss_curve.pdf")
print("内存使用曲线(PNG)已生成: ../../results/charts/fig7_stability.png")
plt.show()
