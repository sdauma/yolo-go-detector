#!/usr/bin/env python3
# 测试字体配置

import matplotlib.pyplot as plt
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

print("测试字体配置...")
# 设置字体（按照《计算机工程》期刊要求）
font_registered = setup_fonts()
print(f"字体注册成功: {font_registered}")

# 打印字体信息
print_font_info()

# 测试绘制一个简单的图表
print("\n测试绘制图表...")
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("测试图表 - Test Chart")
plt.xlabel("X轴 - X Axis")
plt.ylabel("Y轴 - Y Axis")
plt.grid(True)

# 保存测试图表
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "charts")
os.makedirs(output_dir, exist_ok=True)
test_chart_path = os.path.join(output_dir, "test_font_config.png")
plt.savefig(test_chart_path, dpi=600, bbox_inches='tight')
print(f"测试图表已保存到: {test_chart_path}")

plt.close()
print("\n测试完成！")
