#!/usr/bin/env python3
# generate_main_charts.py
# 生成主要的图表文件：YOLO演化趋势、推理流程图和内存管理对比图

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as lines

# 设置中文字体为华文中宋，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['font.family'] = ['sans-serif', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
paper_images_dir = os.path.join(project_root, "results", "charts")

# 确保images目录存在
os.makedirs(paper_images_dir, exist_ok=True)

def generate_yolo_evolution_chart():
    """生成YOLO系列模型参数量与FLOPs演化趋势图表"""
    plt.figure(figsize=(10, 6))
    
    # 数据
    models = ['YOLOv3', 'YOLOv4', 'YOLOv5x', 'YOLO11x']
    params = [62, 70, 87, 230]  # 参数量 (M)
    flops = [65, 75, 90, 220]   # FLOPs (G)
    
    # 创建双轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 参数量轴
    ax1.set_xlabel('YOLO模型版本', fontsize=12)
    ax1.set_ylabel('参数量 (M)', fontsize=12, color='blue')
    ax1.plot(models, params, 'o-', color='blue', linewidth=2, markersize=8, label='参数量 (M)')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 250)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # FLOPs轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('FLOPs (G)', fontsize=12, color='red')
    ax2.plot(models, flops, 's-', color='red', linewidth=2, markersize=8, label='FLOPs (G)')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 250)
    
    # 标题和图例
    plt.title('YOLO系列模型参数量与FLOPs演化趋势', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(paper_images_dir, "yolo_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"YOLO系列模型参数量与FLOPs演化趋势图表已保存到: {output_path}")
    plt.close()

def generate_inference_flow_chart():
    """生成深度学习推理完整流程图"""
    plt.figure(figsize=(12, 6))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # 节点位置
    nodes = [
        (1, 2),  # 输入图像
        (3, 2),  # 图像预处理
        (5, 2),  # 张量构造
        (7, 2),  # ONNX Runtime推理执行
        (9, 2),  # 输出张量
        (7, 1),  # 后处理
        (5, 1),  # 最终结果
    ]
    
    # 节点标签
    node_labels = [
        '输入图像\n(640×640)',
        '图像预处理\n(归一化/Resize)',
        '张量构造\n(NCHW格式)',
        'ONNX Runtime\n推理执行',
        '输出张量\n(检测结果)',
        '后处理\n(NMS/筛选)',
        '最终结果\n(检测框/置信度)',
    ]
    
    # 节点颜色
    node_colors = [
        '#98fb98',  # 绿色
        '#87ceeb',  # 蓝色
        '#ffb6c1',  # 粉色
        '#ff6347',  # 红色
        '#dda0dd',  # 紫色
        '#40e0d0',  # 青色
        '#98fb98',  # 绿色
    ]
    
    # 绘制节点
    rects = []
    for (x, y), label, color in zip(nodes, node_labels, node_colors):
        rect = patches.Rectangle((x-0.8, y-0.5), 1.6, 1.0, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
        rects.append(rect)
    
    # 绘制箭头
    arrows = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[4]),
        (nodes[4], nodes[5]),
        (nodes[5], nodes[6]),
    ]
    
    for (start, end) in arrows:
        arrow = lines.Line2D(
            [start[0]+0.8, end[0]-0.8],
            [start[1], end[1]],
            linewidth=2, color='black', marker='>', markersize=10,
            markerfacecolor='black', markeredgecolor='black'
        )
        ax.add_line(arrow)
    
    # 标注差异区域
    ax.text(7, 3, 'Python/Go差异区域', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(5, 3, '堆内/堆外操作', ha='center', va='center', fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(paper_images_dir, "inference_flow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"深度学习推理完整流程图已保存到: {output_path}")
    plt.close()

def generate_memory_comparison_chart():
    """生成Go与Python语言绑定内存管理机制对比图"""
    plt.figure(figsize=(10, 6))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Go内存管理节点
    go_nodes = [
        (3, 4),  # Go内存分配
        (3, 3),  # 三色标记清除GC
        (3, 2),  # 零拷贝优化
    ]
    
    # Python内存管理节点
    python_nodes = [
        (7, 4),  # Python内存分配
        (7, 3),  # 引用计数管理
        (7, 2),  # 垃圾回收
    ]
    
    # 节点标签
    go_labels = [
        'Go内存分配\n(tcmalloc变种)',
        '三色标记清除GC\n(零漂移)',
        '零拷贝优化\n(直接操作)',
    ]
    
    python_labels = [
        'Python内存分配\n(pymalloc池)',
        '引用计数管理\n(频繁更新)',
        '垃圾回收\n(周期性扫描)',
    ]
    
    # 绘制Go节点
    for (x, y), label in zip(go_nodes, go_labels):
        rect = patches.Rectangle((x-1.2, y-0.4), 2.4, 0.8, linewidth=2, edgecolor='#1f77b4', facecolor='#e3f2fd')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color='#1f77b4')
    
    # 绘制Python节点
    for (x, y), label in zip(python_nodes, python_labels):
        rect = patches.Rectangle((x-1.2, y-0.4), 2.4, 0.8, linewidth=2, edgecolor='#ff7f0e', facecolor='#fff3e0')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color='#ff7f0e')
    
    # 绘制Go箭头
    for i in range(len(go_nodes)-1):
        start = go_nodes[i]
        end = go_nodes[i+1]
        arrow = lines.Line2D(
            [start[0], end[0]],
            [start[1]-0.4, end[1]+0.4],
            linewidth=2, color='#1f77b4', marker='>', markersize=10,
            markerfacecolor='#1f77b4', markeredgecolor='#1f77b4'
        )
        ax.add_line(arrow)
    
    # 绘制Python箭头
    for i in range(len(python_nodes)-1):
        start = python_nodes[i]
        end = python_nodes[i+1]
        arrow = lines.Line2D(
            [start[0], end[0]],
            [start[1]-0.4, end[1]+0.4],
            linewidth=2, color='#ff7f0e', marker='>', markersize=10,
            markerfacecolor='#ff7f0e', markeredgecolor='#ff7f0e'
        )
        ax.add_line(arrow)
    
    # 底部标签
    ax.text(3, 1, 'Go: 稳定内存占用', ha='center', va='center', fontsize=10, color='#1f77b4', fontweight='bold')
    ax.text(7, 1, 'Python: 内存增长趋势', ha='center', va='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(paper_images_dir, "memory_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Go与Python语言绑定内存管理机制对比图表已保存到: {output_path}")
    plt.close()

def main():
    print("===== 开始生成主要图表 ======")
    
    # 生成主要图表
    generate_yolo_evolution_chart()
    generate_inference_flow_chart()
    generate_memory_comparison_chart()
    
    print("\n===== 所有主要图表生成完成！ =====")
    print(f"图表保存位置: {paper_images_dir}")

if __name__ == "__main__":
    main()
