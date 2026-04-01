#!/usr/bin/env python3
# generate_reference_flowchart_detailed.py
# 生成参考文献筛选与核验流程图（超细化版）

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

def generate_reference_flowchart_detailed():
    """生成参考文献筛选与核验流程图（超细化版）"""
    plt.figure(figsize=(12, 14))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 节点定义（从上到下）
    nodes = [
        {'pos': (7, 13), 'label': '文献检索开始\nLiterature Retrieval Initiated', 'width': 3.5, 'height': 0.9, 'type': 'start', 'color': '#90EE90'},
        {'pos': (7, 11.5), 'label': '多数据库文献检索\nMulti-database Literature Search\n\nIEEE Xplore\nACM Digital Library\nSpringerLink\nUSENIX\narXiv\nGoogle Scholar\n官方技术文档', 'width': 4, 'height': 1.2, 'type': 'process', 'color': '#E1F5FE'},
        {'pos': (7, 10), 'label': '去重与初步整理\nDeduplication and Initial Organization\n\n删除重复文献\n统一文献信息格式', 'width': 4, 'height': 1.2, 'type': 'process', 'color': '#E1F5FE'},
        {'pos': (7, 8.5), 'label': '标题与摘要相关性筛选\nTitle and Abstract Relevance Screening\n\n是否涉及深度学习推理\n是否涉及性能评估\n是否涉及推理系统或运行时', 'width': 4, 'height': 1.2, 'type': 'process', 'color': '#E1F5FE'},
        {'pos': (7, 6.8), 'label': '是否与研究主题直接相关？\nDirectly Relevant to Research Topic?', 'width': 3.5, 'height': 1.2, 'type': 'decision', 'color': '#FFD700'},
        {'pos': (11.5, 6.8), 'label': '排除文献\nExcluded Records', 'width': 2.5, 'height': 0.9, 'type': 'exclude', 'color': '#FF6B6B'},
        {'pos': (7, 5.3), 'label': '来源与出版信息核验\nSource and Publication Verification\n\n作者信息完整\n出版来源可查\n年份明确\n数据库可检索', 'width': 4, 'height': 1.2, 'type': 'process', 'color': '#E1F5FE'},
        {'pos': (7, 3.8), 'label': '文献来源可靠且信息完整？\nReliable Source and Complete Metadata?', 'width': 3.5, 'height': 1.2, 'type': 'decision', 'color': '#FFD700'},
        {'pos': (11.5, 3.8), 'label': '排除文献\nExcluded Records', 'width': 2.5, 'height': 0.9, 'type': 'exclude', 'color': '#FF6B6B'},
        {'pos': (7, 2.3), 'label': '全文审阅与方法学价值评估\nFull-text Review and Methodological Assessment\n\n是否支持研究框架\n是否具备引用价值\n是否可复现或可验证', 'width': 4, 'height': 1.2, 'type': 'process', 'color': '#E1F5FE'},
        {'pos': (7, 0.8), 'label': '纳入最终参考文献集合\nIncluded in Final Reference Set', 'width': 3.5, 'height': 0.9, 'type': 'start', 'color': '#90EE90'},
    ]
    
    # 绘制节点
    for node in nodes:
        x, y = node['pos']
        width = node['width']
        height = node['height']
        label = node['label']
        node_type = node['type']
        color = node['color']
        
        if node_type == 'start' or node_type == 'exclude':
            # 椭圆节点
            ellipse = patches.Ellipse(
                (x, y), width/2, height/2,
                linewidth=2, edgecolor='black', facecolor=color
            )
            ax.add_patch(ellipse)
            ax.text(x, y, label, ha='center', va='center', fontsize=8)
        elif node_type == 'process':
            # 矩形节点
            rect = patches.Rectangle(
                (x - width/2, y - height/2),
                width, height,
                linewidth=2, edgecolor='black', facecolor=color
            )
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=8)
        elif node_type == 'decision':
            # 菱形节点
            diamond = patches.Polygon([
                (x, y + height/2),
                (x + width/2, y),
                (x, y - height/2),
                (x - width/2, y)
            ], linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(diamond)
            ax.text(x, y, label, ha='center', va='center', fontsize=7)
    
    # 绘制箭头（主流程）
    main_arrows = [
        (0, 1),  # 开始 → 检索
        (1, 2),  # 检索 → 去重
        (2, 3),  # 去重 → 筛选
        (3, 4),  # 筛选 → 决策1
        (4, 6),  # 决策1 YES → 核验
        (6, 7),  # 核验 → 决策2
        (7, 9),  # 决策2 YES → 评估
        (9, 10), # 评估 → 终点
    ]
    
    for (start_idx, end_idx) in main_arrows:
        start = nodes[start_idx]
        end = nodes[end_idx]
        
        # 箭头从上一个节点底部到下一个节点顶部
        start_y = start['pos'][1] - start['height']/2
        end_y = end['pos'][1] + end['height']/2
        
        arrow = lines.Line2D(
            [start['pos'][0], end['pos'][0]],
            [start_y, end_y],
            linewidth=2, color='black', marker='v', markersize=12,
            markerfacecolor='black', markeredgecolor='black'
        )
        ax.add_line(arrow)
    
    # 绘制排除箭头（虚线）
    exclude_arrows = [
        (4, 5),  # 决策1 NO → 排除1
        (7, 8),  # 决策2 NO → 排除2
    ]
    
    for (start_idx, end_idx) in exclude_arrows:
        start = nodes[start_idx]
        end = nodes[end_idx]
        
        # 箭头从决策节点右侧到排除节点
        start_y = start['pos'][1]
        end_y = end['pos'][1]
        
        arrow = lines.Line2D(
            [start['pos'][0] + start['width']/2, end['pos'][0] - end['width']/2],
            [start_y, end_y],
            linewidth=1.5, color='black', linestyle='--', marker='>', markersize=10,
            markerfacecolor='black', markeredgecolor='black'
        )
        ax.add_line(arrow)
    
    # 添加决策分支标注
    ax.text(nodes[4]['pos'][0] + 1.5, nodes[4]['pos'][1] + 0.3, 'YES', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(nodes[4]['pos'][0] - 1.5, nodes[4]['pos'][1] + 0.3, 'NO', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(nodes[7]['pos'][0] + 1.5, nodes[7]['pos'][1] + 0.3, 'YES', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(nodes[7]['pos'][0] - 1.5, nodes[7]['pos'][1] + 0.3, 'NO', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存PNG格式（600dpi）
    output_png = os.path.join(paper_images_dir, "reference_flowchart_detailed.png")
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    print(f"参考文献筛选与核验流程图（PNG）已保存到: {output_png}")
    
    # 保存PDF格式
    output_pdf = os.path.join(paper_images_dir, "reference_flowchart_detailed.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"参考文献筛选与核验流程图（PDF）已保存到: {output_pdf}")
    
    plt.close()

def main():
    print("===== 开始生成参考文献筛选与核验流程图（超细化版） =====")
    
    # 生成流程图
    generate_reference_flowchart_detailed()
    
    print("\n===== 参考文献筛选与核验流程图生成完成！ =====")
    print(f"图表保存位置: {paper_images_dir}")

if __name__ == "__main__":
    main()
