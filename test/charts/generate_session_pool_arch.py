import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import font_manager
import os

# 注册华文中宋字体（检查多个路径）
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 字体文件可能存在的路径
font_paths = [
    os.path.join(base_dir, "paper", "华文中宋.ttf"),  # 项目目录
    "C:\\Windows\\Fonts\\华文中宋.ttf",  # 系统字体目录
    "C:\\Users\\Administrator\\AppData\\Local\\Microsoft\\Windows\\Fonts\\华文中宋.ttf"  # 用户字体目录
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

# 设置字体：中文使用华文中宋，英文使用 Times New Roman
plt.rcParams['font.sans-serif'] = ['STZhongsong', 'SimHei']  # 优先华文中宋，回退到黑体
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

fig, ax = plt.subplots(figsize=(12, 7))  # 增大画布以容纳更详细的标注

# 绘制函数
def box(x, y, w, h, text, fontsize=10):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.5, edgecolor='black')
    ax.add_patch(rect)
    # 不指定 family，让 matplotlib 自动根据字符选择字体
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', linewidth=1.2, edgecolor='black')
    ax.add_patch(arr)

# 摄像头
box(0.5, 3.5, 1.5, 0.8, "摄像头流\n2600 路", fontsize=9)

# 任务队列
box(3, 3.5, 1.8, 0.8, "推理任务队列", fontsize=9)

# Session Pool 管理器（增强标注）
box(6, 3.2, 2.2, 1.4, "Session Pool 管理器\n(CPU 核心数配置)\n轮询调度", fontsize=9)

# 多个 session
box(9.2, 4.2, 1.4, 0.6, "Session 1", fontsize=9)
box(9.2, 3.3, 1.4, 0.6, "Session 2", fontsize=9)
box(9.2, 2.4, 1.4, 0.6, "Session N", fontsize=9)

# 检测结果
box(11.5, 3.5, 1.8, 0.8, "检测结果\n输出", fontsize=9)

# 箭头
arrow(2, 3.9, 3, 3.9)
arrow(4.8, 3.9, 6, 3.9)

arrow(8.2, 4.0, 9.2, 4.5)
arrow(8.2, 3.9, 9.2, 3.6)
arrow(8.2, 3.8, 9.2, 2.7)

arrow(10.6, 4.5, 11.5, 3.9)
arrow(10.6, 3.6, 11.5, 3.9)
arrow(10.6, 2.7, 11.5, 3.9)

# 设置坐标轴
ax.set_xlim(0, 14)
ax.set_ylim(2, 5.5)
ax.axis('off')

# 添加中英文标题（符合期刊图注规范）
# 使用 suptitle 并让 matplotlib 自动处理字体
plt.suptitle("图 1 Session Pool 并发推理架构\nFig. 1 Session Pool Concurrent Inference Architecture",
             fontsize=12, y=0.98)

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
plt.savefig(os.path.join(final_charts_dir, "fig1_session_pool_architecture.png"), dpi=600, bbox_inches='tight', format='png')
print("图 1 已生成：fig1_session_pool_architecture.png")
print("图表已保存到：", output_dir)
print("最终图表已保存到：", final_charts_dir)
plt.close()