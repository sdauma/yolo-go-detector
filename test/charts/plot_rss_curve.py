import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*iCCP.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

# 与论文其余图表统一：用 font_utils 注册华文中宋（回退宋体/黑体），避免字体不一致
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import font_utils
font_utils.setup_fonts()
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取已经生成的 CSV
try:
    go_rss = pd.read_csv("../../results/go_rss_curve.csv")
    if go_rss.empty:
        raise ValueError("Go 内存数据为空")
    go_mem_col = "RSS_MB" if "RSS_MB" in go_rss.columns else "PM_MB"
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

plt.plot(go_time_min, go_rss[go_mem_col], 'k--o', label="Go", linewidth=1.5, markersize=4)
py_mem_col = "RSS_MB" if "RSS_MB" in py_rss.columns else "PM_MB"
plt.plot(py_time_min, py_rss[py_mem_col], 'k-.s', label="Python", linewidth=1.5, markersize=4)

plt.xlabel("运行时间 (min)", fontsize=11)
plt.ylabel("内存占用 (MB)", fontsize=11)
plt.title("长时间运行的内存漂移对比", fontsize=12)
plt.legend(fontsize=10)
plt.grid(linestyle=":", linewidth=0.5, color='gray', alpha=0.6)

plt.tight_layout()
plt.savefig("../../results/rss_curve.pdf", dpi=600)
plt.savefig("../../results/charts/rss_curve.png", dpi=600)
print("内存使用曲线已生成: ../../results/rss_curve.pdf")
print("内存使用曲线(PNG)已生成: ../../results/charts/rss_curve.png")

