#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 paper_final.tex 引用的 Arena 核心发现双面板图（fig8_arena_sweetspot）。

左面板 : Arena ON/OFF 的 PM 漂移对比（表9，§4.6.1）。
          Go Session Pool 采用 §4.6.2 租用模式稳态值 794.35/10.52 MB，
          而非表9固定分配模式的负值假象（见表9注）。
右面板 : 甜点效应（表10，§4.6.2），并发 Run()=4 固定。

风格与 test/charts 下其他作图脚本保持一致：
  - 字体：华文中宋(STZhongsong) + Times New Roman
  - 配色：Arena ON #E8E8E8（浅灰）、Arena OFF #B8B8B8（中灰，加 hatch）
  - 输出：results/charts/fig8_arena_sweetspot.png + .pdf (dpi=600)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 字体与论文主图生成脚本(generate_all_charts.py)保持一致：
# 用 font_utils 显式注册华文中宋，避免依赖字体缓存、比裸 rcParams 更稳健。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import font_utils
font_utils.setup_fonts()

results_dir = '../../results'

C_ON = '#E8E8E8'    # Arena ON（浅灰）
C_OFF = '#B8B8B8'   # Arena OFF（中灰）
EDGE = 'black'
LW = 0.5

# ---------- 左面板：Arena ON/OFF 漂移对比（表9，§4.6.1） ----------
labels = ['Go\nUnsafe Shared', 'Go\nSession Pool*',
          'Python\nUnsafe Shared', 'Python\nSession Pool']
on = [2169.38, 794.35, 2126.86, 2154.21]
off = [235.26, 10.52, 243.17, 916.96]

x = np.arange(len(labels))
w = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

bars1 = ax1.bar(x - w / 2, on, w, label='Arena ON', color=C_ON,
                edgecolor=EDGE, linewidth=LW)
bars2 = ax1.bar(x + w / 2, off, w, label='Arena OFF', color=C_OFF,
                edgecolor=EDGE, linewidth=LW, hatch='//')
ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('PM 漂移 (MB, 对数轴)', fontsize=10)
ax1.set_title('(a) Arena ON/OFF 漂移对比', fontsize=11)
ax1.legend(fontsize=9, loc='best')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for b in list(bars1) + list(bars2):
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width() / 2, h * 1.07, f'{h:.1f}',
             ha='center', va='bottom', fontsize=8)
ax1.text(0.02, 0.02,
         '*Go Session Pool 为租用模式(§4.6.2)\n非表9固定分配负值假象',
         transform=ax1.transAxes, fontsize=8, va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   alpha=1.0, edgecolor='black', linewidth=0.5))

# ---------- 右面板：甜点效应（表10，§4.6.2） ----------
sessions = [1, 4, 6, 8, 12, 16]
on_s = [1883.30, 794.35, 1587.31, 2116.46, 3174.34, 4232.51]
off_s = [15.68, 10.52, 12.72, 12.26, 1.71, 1.49]

# Arena ON 线改用深灰 #666666（C_ON=#E8E8E8 仅作柱填充，作折线在白底/B&W 下对比度过低会消失）
ax2.plot(sessions, on_s, 'o-', color='#666666', label='Arena ON', linewidth=2.0)
ax2.plot(sessions, off_s, 's--', color=C_OFF, label='Arena OFF', linewidth=1.5)
ax2.set_yscale('log')
ax2.set_xlabel('Session 数 (池容量)', fontsize=10)
ax2.set_ylabel('推理阶段漂移 (MB, 对数轴)', fontsize=10)
ax2.set_title('(b) 甜点效应 (并发Run()=4)', fontsize=11)
ax2.axvline(4, color='gray', ls=':', lw=1)
ax2.text(4.3, ax2.get_ylim()[1] * 0.92, '甜点\nN=C=4', fontsize=8, va='top')
ax2.legend(fontsize=9, loc='best')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for xs, ys in ((sessions, on_s), (sessions, off_s)):
    for xx, yy in zip(xs, ys):
        ax2.text(xx, yy * 1.06, f'{yy:.1f}', fontsize=8, ha='center', va='bottom')

# fig.suptitle('Arena 机制核心发现：漂移因果与甜点效应', fontsize=13, y=1.02)  # 图注由论文 \caption 提供，与 fig1–fig7 惯例保持一致
fig.tight_layout()

out_png = os.path.join(results_dir, 'charts', 'fig8_arena_sweetspot.png')
out_pdf = os.path.join(results_dir, 'charts', 'fig8_arena_sweetspot.pdf')
fig.savefig(out_png, dpi=600, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'已保存: {out_png}')
print(f'已保存: {out_pdf}')
