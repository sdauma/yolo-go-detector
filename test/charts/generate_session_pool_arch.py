# -*- coding: utf-8 -*-
"""
图1生成脚本：基于三层结构的 Session Pool 并发推理架构图

本脚本生成论文图1，展示YOLO目标检测系统中Go语言并发推理的分层架构：
  第一层：摄像头流数据输入（2600路视频流）
  第二层：推理任务队列（有界缓冲，削峰填谷）
  第三层：Worker Pool（工作协程池，从队列消费任务）
  第四层：Session Pool（ONNX会话池，Worker通过GetSession获取Session进行推理）
  第五层：检测结果输出

数据流：摄像头 → 任务队列 → Worker Pool → Session Pool → 检测结果输出
核心机制：Session Pool通过池化复用ONNX Runtime Session，避免频繁创建/销毁的开销
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
import os

# 注册华文中宋字体（检查多个路径）
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 字体文件可能存在的路径
font_paths = [
    "C:\\Windows\\Fonts\\simsun.ttc",  # 系统宋体
    "C:\\Users\\Administrator\\AppData\\Local\\Microsoft\\Windows\\Fonts\\simsun.ttc"  # 用户字体目录
]

# 打印当前工作目录
print("Current working directory:", os.getcwd())

# 尝试注册字体
font_registered = False
for font_path in font_paths:
    print("Checking font path:", font_path)
    print("Font file exists:", os.path.exists(font_path))
    if os.path.exists(font_path):
        # 添加到字体管理器
        font_manager.fontManager.addfont(font_path)
        print("Successfully registered font:", font_path)
        font_registered = True
        break

if not font_registered:
    print("Warning: Font file not found in any path, will use system default Chinese font")

# 检查字体是否已注册
font_names = [f.name for f in font_manager.fontManager.ttflist]
print("Registered fonts containing 'STZhongsong':", [name for name in font_names if 'STZhongsong' in name])

# 设置字体：中文使用宋体，英文使用 Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei']  # 优先宋体，回退到黑体
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文使用 Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 打印最终使用的字体配置
print("\n=== 字体配置信息 ===")
print("中文字体 (sans-serif):", plt.rcParams['font.sans-serif'])
print("英文字体 (serif):", plt.rcParams['font.serif'])

# 检查实际可用的中文字体
available_chinese_fonts = [name for name in font_names if 'STZhongsong' in name or 'SimHei' in name or 'SimSun' in name]
print("可用的中文字体:", available_chinese_fonts[:5] if available_chinese_fonts else "未找到特定中文字体")

# 检查实际可用的英文字体
available_english_fonts = [name for name in font_names if 'Times' in name]
print("可用的Times字体:", available_english_fonts[:5] if available_english_fonts else "未找到Times New Roman")
print("===================\n")

# 定义路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(base_dir, "results", "charts")
final_charts_dir = os.path.join(output_dir, "final_charts")

# 创建目录
os.makedirs(final_charts_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))  # 加宽画布以容纳 Worker Pool

# 绘制函数
def box(x, y, w, h, text, fontsize=10, fill=False, facecolor='lightgray'):
    rect = Rectangle((x, y), w, h, fill=fill, linewidth=1.5, edgecolor='black',
                     facecolor=facecolor, alpha=0.3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

def dashed_box(x, y, w, h, text, fontsize=10):
    """虚线框，用于表示逻辑分组"""
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.0, edgecolor='gray',
                     linestyle='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color='gray', style='italic')

def arrow(x1, y1, x2, y2, style='->', color='black'):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, linewidth=1.2,
                          edgecolor=color, mutation_scale=18)
    ax.add_patch(arr)

# === 第一层：数据输入 ===
box(0.3, 4.8, 1.5, 0.8, "摄像头流\n(2600 路)", fontsize=9)

# === 第二层：任务队列（削峰填谷）===
box(2.5, 4.8, 1.8, 0.8, "推理任务队列\n(有界缓冲)", fontsize=9)

# === 第三层：Worker Pool（工作协程池）===
# 虚线大框表示 Worker Pool 逻辑分组
dashed_box(5.0, 3.8, 3.6, 2.2, "工作协程池\n(Worker Pool)", fontsize=9)
# 多个 Worker 协程（灰度填充，兼容黑白印刷）
box(5.3, 5.0, 1.2, 0.6, "Worker 1", fontsize=8, fill=True, facecolor='#E8E8E8')
box(6.8, 5.0, 1.2, 0.6, "Worker 2", fontsize=8, fill=True, facecolor='#E8E8E8')
box(5.3, 4.0, 1.2, 0.6, "Worker 3", fontsize=8, fill=True, facecolor='#E8E8E8')
box(6.8, 4.0, 1.2, 0.6, "Worker N", fontsize=8, fill=True, facecolor='#E8E8E8')

# === 第四层：Session Pool（大框内嵌 Session 子框，展示池内结构）===
# Session Pool 虚线框（右移至 x=9.4 与 Worker Pool 拉开间距给 GetSession 标注留空间；高度缩至 2.2 与 Worker Pool 顶部对齐）
dashed_box(9.4, 3.8, 3.6, 2.2, "", fontsize=9)
# 标题文字放在框内左侧（左对齐）
ax.text(9.6, 5.1, "Session Pool\n(ONNX 会话池)\nSession 级\n轮询分配", fontsize=8,
        ha='left', va='center', color='gray', style='italic')
# Session 子框紧挨文字右侧
box(11.4, 5.3, 1.2, 0.55, "Session 1", fontsize=8)
box(11.4, 4.65, 1.2, 0.55, "Session 2", fontsize=8)
box(11.4, 4.0, 1.2, 0.55, "Session N", fontsize=8)

# === 第五层：检测结果输出 ===
box(13.4, 4.8, 1.6, 0.8, "检测结果\n输出", fontsize=9)

# === 箭头 ===
# 摄像头 → 任务队列
arrow(1.8, 5.2, 2.5, 5.2)

# 任务队列 → Worker Pool（整体）
arrow(4.3, 5.2, 5.0, 5.2)
# 任务队列 → Worker Pool 内部各 Worker 分发（起点统一对齐到任务队列右边缘中心）
arrow(4.3, 5.2, 5.3, 5.3)
arrow(4.3, 5.2, 6.8, 5.3)
arrow(4.3, 5.2, 5.3, 4.3)
arrow(4.3, 5.2, 6.8, 4.3)

# Worker Pool → Session Pool (GetSession)
arrow(8.6, 5.3, 9.4, 5.3)
# 标注获取接口（位于两框中间，间距 0.8 足够文字不落在框边上）
ax.text(9.0, 5.45, "GetSession", fontsize=7, ha='center', va='bottom', style='italic')

# Session 子框 → 检测结果输出（数据流从 Session Pool 内的 Session 汇聚到输出）
arrow(12.6, 5.57, 13.4, 5.2)
arrow(12.6, 4.92, 13.4, 5.2)
arrow(12.6, 4.27, 13.4, 5.2)

# === 设置坐标轴 ===
ax.set_xlim(0, 16.0)
ax.set_ylim(3.2, 6.2)
ax.axis('off')

# 注：图内不放标题，图注由论文 LaTeX \\caption 提供（符合《计算机系统应用》规范）

# 检测实际使用的字体
from matplotlib.font_manager import findfont, FontProperties

# 检测中文字体使用的字体
chinese_text = "摄像头流"
english_text = "Session"

# 获取实际使用的字体文件路径
chinese_font_path = findfont(FontProperties(family=plt.rcParams['font.sans-serif']))
english_font_path = findfont(FontProperties(family=plt.rcParams['font.serif']))

print("\n=== 实际使用的字体 ===")
print("中文字体文件路径:", chinese_font_path)
print("英文字体文件路径:", english_font_path)

# 从路径中提取字体名称
chinese_font_name = os.path.basename(chinese_font_path).lower()
english_font_name = os.path.basename(english_font_path).lower()

if 'stzhongsong' in chinese_font_name or 'huawen' in chinese_font_name or '华文中宋' in chinese_font_path:
    print("[OK] 中文实际使用: 华文中宋 (STZhongsong)")
elif 'simhei' in chinese_font_name:
    print("[OK] 中文实际使用: 黑体 (SimHei) - 回退字体")
elif 'simsun' in chinese_font_name:
    print("[OK] 中文实际使用: 宋体 (SimSun) - 回退字体")
else:
    print("[?] 中文实际使用:", os.path.basename(chinese_font_path))

if 'times' in english_font_name:
    print("[OK] 英文实际使用: Times New Roman")
else:
    print("[?] 英文实际使用:", os.path.basename(english_font_path))
print("=====================\n")

# 保存图片
plt.savefig(os.path.join(output_dir, "fig1_session_pool_architecture.png"), dpi=600, bbox_inches='tight', format='png')
plt.savefig(os.path.join(output_dir, "fig1_session_pool_architecture.pdf"), bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(final_charts_dir, "fig1_session_pool_architecture.png"), dpi=600, bbox_inches='tight', format='png')
print("图 1 已生成：fig1_session_pool_architecture.png")
print("图表已保存到：", output_dir)
print("最终图表已保存到：", final_charts_dir)
plt.close()