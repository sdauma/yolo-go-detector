# -*- coding: utf-8 -*-
r"""
图1生成脚本：基于三层结构的 Session Pool 并发推理架构图

期刊级信息架构（参考 ChatGPT 版布局优点，保持 matplotlib 矢量轻量本质）：
  - 扁宽布局：主流程从左至右一条直线，三层核心（任务队列/Worker Pool/Session Pool）突出
  - Worker 2×2 紧凑、Session 单列居中
  - 不绘制图例（符号为通用惯例，含义由 caption/正文承载，符合核心期刊规范）
  - GetSession（实线）/ PutSession（虚线返回）语义明确
  - 不保留图内大段说明框：解释性文字由 LaTeX \caption 与正文 §2.2.3 承载，符合核心期刊规范
  - 术语严格对齐论文 §2.2.3 与代码：任务队列 / 工作协程池 / Session Pool（会话池）/
    GetSession / PutSession；不引入"Worker Pool""ONNX 会话池""轮询分配"等论文未用措辞
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*iCCP.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import os
import dataclasses

# ============================================================
# 字体注册（优先 STZhongsong，失败回退 SimSun）
# ============================================================
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 统一使用 ASCII 别名，避免某些 .ttf 内部名含中文导致 PDF 后端
# UnicodeEncodeError: 'ascii' codec can't encode characters
CN_FONT_ALIAS = "STZhongsong"

font_paths = [
    os.path.join(base_dir, "paper", "STZhongsong.ttf"),  # 英文文件名副本：华文中宋内部名 STZhongsong(ASCII)
                                                         # 用英文名避免 matplotlib 以含中文的文件名派生 PDF 字体资源名
    os.path.join(base_dir, "paper", "华文中宋.ttf"),      # 原始中文文件名（兜底）
    "C:\\Windows\\Fonts\\STZHONGS.TTF",                  # 系统华文中宋（回退）
    "C:\\Windows\\Fonts\\simsun.ttc",                    # 宋体（回退）
]

registered_path = None
for font_path in font_paths:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        registered_path = font_path
        # 把该字体的内部名（可能是中文）覆盖为 ASCII 别名，
        # 否则 matplotlib 写 PDF 时字体资源名含中文会触发 ascii 编码错误
        # 注意：FontEntry 是冻结 dataclass，需用 dataclasses.replace 生成新条目
        norm = os.path.normcase(font_path)
        ttflist = font_manager.fontManager.ttflist
        for i, f in enumerate(ttflist):
            if os.path.normcase(f.fname) == norm:
                ttflist[i] = dataclasses.replace(f, name=CN_FONT_ALIAS)
        break

if registered_path:
    print("中文字体已加载：", registered_path)
else:
    print("Warning: No Chinese font found, using system default")

# 全局字体设置：中文 STZhongsong，英文通过 FontProperties 单独指定 Times New Roman
plt.rcParams['font.sans-serif'] = [CN_FONT_ALIAS, 'SimSun', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 纯英文标签使用 Times New Roman
font_en = FontProperties(family='Times New Roman', size=10)

# ============================================================
# 路径与画布（扁宽布局，高度加大给主图留足空间）
# ============================================================
output_dir = os.path.join(base_dir, "results", "charts")
final_charts_dir = os.path.join(output_dir, "final_charts")
os.makedirs(final_charts_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 6.5))

# ============================================================
# 绘制函数
# ============================================================
def box(x, y, w, h, text, fontsize=11, fill=True, facecolor='#E8E8E8', fp=None):
    """实线框，默认浅灰填充（兼容黑白印刷）。fp 用于纯英文标签。"""
    rect = Rectangle((x, y), w, h, fill=fill, linewidth=1.2, edgecolor='black',
                     facecolor=facecolor, alpha=0.35 if fill else 0)
    ax.add_patch(rect)
    if fp is not None:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontproperties=fp)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

def dashed_box(x, y, w, h, title, fontsize=10):
    """虚线框（逻辑分组），标题置于顶部居中，与内部元素保持充足间距"""
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.0, edgecolor='gray',
                     linestyle='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.18, title, ha='center', va='top', fontsize=fontsize,
            color='black')

def arrow(x1, y1, x2, y2, color='black', lw=1.2, style='->', ls='-'):
    """实线/虚线箭头"""
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, linewidth=lw,
                          edgecolor=color, mutation_scale=16, linestyle=ls)
    ax.add_patch(arr)

# ============================================================
# 主流程框（统一中心 Y=3.2，高度 1.0）
# ============================================================
Y = 3.2
H = 1.0

# 输入边界
box(0.4, Y - H/2, 1.8, H, "视频流\n输入", fontsize=11)

# 第一层：任务队列（论文 §2.2.3 原文"任务队列——有界缓冲"）
box(2.8, Y - H/2, 2.0, H, "任务队列", fontsize=11)
# "有界缓冲"作为框下小注，避免框内括号堆叠
ax.text(2.8 + 1.0, Y - H/2 - 0.22, "（有界缓冲）", ha='center', va='top',
        fontsize=9, color='gray')

# 输出边界（右移，与 Session Pool 之间留出空隙放置轮询分配标签）
box(15.0, Y - H/2, 1.8, H, "检测结果\n输出", fontsize=11)

# ============================================================
# 第二层：Worker Pool（虚线框 + 内部 2×2 Worker）
# ============================================================
WP_X, WP_W = 5.4, 4.0
WP_Y, WP_H = 1.4, 3.0
dashed_box(WP_X, WP_Y, WP_W, WP_H, "工作 goroutine 池", fontsize=10)

worker_w, worker_h = 1.35, 0.70
wx1 = WP_X + 0.55
wx2 = WP_X + 2.10
wy_top = 3.20
wy_bot = 1.95

box(wx1, wy_top, worker_w, worker_h, "Worker 1", fill=True, fp=font_en)
box(wx2, wy_top, worker_w, worker_h, "Worker 2", fill=True, fp=font_en)
box(wx1, wy_bot, worker_w, worker_h, "Worker 3", fill=True, fp=font_en)
box(wx2, wy_bot, worker_w, worker_h, "... Worker N", fill=True, fp=font_en)

# ============================================================
# 第三层：Session Pool（虚线框 + 内部 Session 单列居中）
# ============================================================
SP_X, SP_W = 11.0, 3.4
SP_Y, SP_H = 1.4, 3.0
dashed_box(SP_X, SP_Y, SP_W, SP_H, "Session Pool", fontsize=10)
# 论文 §2.2.3 (1) 明确"每个拥有独立 Arena"，补一行注释级小注保证图-文一致
ax.text(SP_X + SP_W/2, SP_Y + SP_H - 0.42, "每 Session 独立 Arena",
        ha='center', va='top', fontsize=9, color='black')

sess_w, sess_h = 1.7, 0.60
sx = SP_X + (SP_W - sess_w) / 2
sy_top = 3.20
sy_mid = 2.45
sy_bot = 1.70

box(sx, sy_top, sess_w, sess_h, "Session 1", fill=True, fp=font_en)
box(sx, sy_mid, sess_w, sess_h, "Session 2", fill=True, fp=font_en)
box(sx, sy_bot, sess_w, sess_h, "... Session N", fill=True, fp=font_en)

# ============================================================
# 箭头（语义区分：实线=数据流，虚线=资源归还，弧线=轮询）
# ============================================================

# 数据流：摄像头 → 队列
arrow(2.2, Y, 2.8, Y, lw=1.2)

# 数据流：队列 → Worker Pool
arrow(4.8, Y, WP_X, Y, lw=1.2)

# 数据流 + 控制调用：Worker Pool → Session Pool（GetSession，加粗主箭头）
get_y = 3.45
arrow(WP_X + WP_W, get_y, SP_X, get_y, lw=1.6)
ax.text((WP_X + WP_W + SP_X)/2, get_y + 0.18, "GetSession",
        fontproperties=font_en, ha='center', va='bottom')

# 资源归还：Session Pool → Worker Pool（PutSession，虚线返回）
put_y = 2.35
arrow(SP_X, put_y, WP_X + WP_W, put_y, lw=1.1, ls='--')
ax.text((WP_X + WP_W + SP_X)/2, put_y - 0.15, "PutSession",
        fontproperties=font_en, ha='center', va='top')

# 数据流：Session Pool → 检测结果输出
arrow(SP_X + SP_W, Y, 15.0, Y, lw=1.2)

# 注：论文 §2.2.3 仅描述"Goroutine 通过 GetSession/PutSession 租用 Session"，
# 未规定 Session 之间的轮询/分配策略，故图中不额外绘制"轮询分配"弧线，避免引入论文未用机制。

# ============================================================
# 图例：不绘制
# 理由：实线框=处理单元、虚线框=逻辑分组、实线箭头=数据流、虚线箭头=资源归还
#       均为通用绘图惯例，审稿人可直接识别；PutSession 已在虚线箭头上以行内标注，
#       无需底部图例重复解释。符号语义由 LaTeX \caption 与正文 §2.2.3 承载，符合
#       中文核心期刊架构图惯例，并避免图例挤占高度导致 LaTeX 排版被过度压缩。
# ============================================================

# ============================================================
# 坐标轴与保存
# （图标题由 LaTeX \caption 生成，图内不硬编码，避免排版重复）
# ============================================================
ax.set_xlim(0, 17.2)
# 收紧 y 范围以贴合内容（池子虚线框 1.4–4.4），避免 axis('off') 下
# bbox_inches='tight' 仍按整个 axes 矩形裁切导致图下方出现大片白边。
ax.set_ylim(1.25, 4.55)
ax.axis('off')

plt.savefig(os.path.join(output_dir, "fig1_session_pool_architecture.png"),
            dpi=600, bbox_inches='tight', format='png')
plt.savefig(os.path.join(output_dir, "fig1_session_pool_architecture.pdf"),
            bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(final_charts_dir, "fig1_session_pool_architecture.png"),
            dpi=600, bbox_inches='tight', format='png')
print("图 1 已生成：fig1_session_pool_architecture.png")
print("图表已保存到：", output_dir)
print("最终图表已保存到：", final_charts_dir)
plt.close()
