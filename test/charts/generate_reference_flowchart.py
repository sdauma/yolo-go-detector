#!/usr/bin/env python3
# generate_reference_flowchart.py
# 生成参考文献筛选与核验流程图（中文习惯版）

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 获取项目根目录
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
paper_images_dir = os.path.join(project_root, "results", "charts")

# 确保images目录存在
os.makedirs(paper_images_dir, exist_ok=True)

def generate_reference_flowchart():
    """生成参考文献筛选与核验流程图（中文习惯版）"""
    plt.figure(figsize=(10, 10))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 节点定义（从上到下）
    nodes = [
        {'pos': (6, 9), 'label': '开始文献筛选', 'width': 3, 'height': 0.8, 'type': 'rect'},
        {'pos': (6, 7.8), 'label': '步骤1：基于关键词进行初始文献检索', 'width': 4, 'height': 0.8, 'type': 'rect'},
        {'pos': (6, 6.4), 'label': '步骤2：依据标题与摘要进行主题相关性筛选', 'width': 4, 'height': 0.8, 'type': 'rect'},
        {'pos': (6, 5), 'label': '步骤3：核验文献来源及出版信息的真实性', 'width': 4, 'height': 0.8, 'type': 'rect'},
        {'pos': (6, 3.6), 'label': '步骤4：确认文献信息完整性与可引用性', 'width': 4, 'height': 0.8, 'type': 'rect'},
        {'pos': (6, 2.2), 'label': '步骤5：形成最终参考文献集合', 'width': 3, 'height': 0.8, 'type': 'rect'},
    ]
    
    # 绘制节点
    for node in nodes:
        x, y = node['pos']
        width = node['width']
        height = node['height']
        label = node['label']
        
        # 绘制矩形节点（白色背景，黑色边框）
        rect = patches.Rectangle(
            (x - width/2, y - height/2),
            width, height,
            linewidth=2, edgecolor='black', facecolor='white'
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9)
    
    # 绘制箭头（从上到下）
    for i in range(len(nodes) - 1):
        start = nodes[i]['pos']
        end = nodes[i + 1]['pos']
        
        # 箭头从上一个节点底部到下一个节点顶部
        start_y = start[1] - nodes[i]['height']/2
        end_y = end[1] + nodes[i + 1]['height']/2
        
        # 绘制实线箭头
        arrow = lines.Line2D(
            [start[0], end[0]],
            [start_y, end_y],
            linewidth=2, color='black', marker='>', markersize=12,
            markerfacecolor='black', markeredgecolor='black'
        )
        ax.add_line(arrow)
    
    # 添加标注（右侧）
    annotations = [
        {'pos': (10.5, 6.4), 'label': '筛选标准：\n与研究主题\n直接相关', 'target': 2},
        {'pos': (10.5, 5), 'label': '核验标准：\n权威数据库或\n官方文档', 'target': 3},
        {'pos': (10.5, 3.6), 'label': '完整性标准：\n作者、标题、\n年份、来源', 'target': 4},
    ]
    
    for ann in annotations:
        x, y = ann['pos']
        label = ann['label']
        target_idx = ann['target']
        
        # 绘制标注框（灰色边框，浅灰背景）
        rect = patches.Rectangle(
            (x - 1.5, y - 0.7),
            3.0, 1.4,
            linewidth=1, edgecolor='gray', facecolor='#f5f5f5',
            linestyle='-'
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=8)
        
        # 绘制虚线箭头（从标注框左侧指向目标节点右侧）
        target_node = nodes[target_idx]
        arrow = lines.Line2D(
            [x - 1.5, target_node['pos'][0] + target_node['width']/2],
            [y, target_node['pos'][1]],
            linewidth=1, color='gray', linestyle='--', marker='>', markersize=8,
            markerfacecolor='gray', markeredgecolor='gray'
        )
        ax.add_line(arrow)
    
    plt.tight_layout()
    
    # 保存PNG格式（600dpi）
    output_png = os.path.join(paper_images_dir, "reference_flowchart.png")
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    print(f"参考文献筛选与核验流程图（PNG）已保存到: {output_png}")
    
    # 保存PDF格式
    output_pdf = os.path.join(paper_images_dir, "reference_flowchart.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"参考文献筛选与核验流程图（PDF）已保存到: {output_pdf}")
    
    plt.close()

def main():
    print("===== 开始生成参考文献筛选与核验流程图（中文习惯版） =====")
    
    # 生成流程图
    generate_reference_flowchart()
    
    print("\n===== 参考文献筛选与核验流程图生成完成！ =====")
    print(f"图表保存位置: {paper_images_dir}")

if __name__ == "__main__":
    main()
