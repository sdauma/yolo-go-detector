import pandas as pd
import matplotlib.pyplot as plt

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
plt.show()
