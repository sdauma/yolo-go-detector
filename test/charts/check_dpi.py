from PIL import Image
import os

# 定义路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
charts_dir = os.path.join(base_dir, "results", "charts")

# 检查的图表文件
chart_files = [
    "fig2_throughput_comparison.png",
    "fig3_memory_comparison.png",
    "fig4_batch_effect.png",
    "fig5_model_size_comparison.png",
    "fig6_cpu_utilization.png",
    "fig7_stability.png"
]

print("=" * 60)
print("检查图表 DPI 分辨率")
print("=" * 60)

for chart_file in chart_files:
    chart_path = os.path.join(charts_dir, chart_file)
    if os.path.exists(chart_path):
        with Image.open(chart_path) as img:
            dpi = img.info.get('dpi', (0, 0))
            print(f"{chart_file}: {dpi[0]:.0f} DPI x {dpi[1]:.0f} DPI")
    else:
        print(f"{chart_file}: 文件不存在")

print("=" * 60)