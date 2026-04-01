#!/usr/bin/env python3
"""
统计41个测试程序的实际推理次数 - 改进版
通过读取文件内容中的注释和关键变量来统计
"""

import os
import re
from pathlib import Path

def count_inferences_in_go_file(filepath):
    """从Go文件中提取推理次数配置"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找 runs := 200 // 每轮200次推理
    runs_match = re.search(r'runs\s*:=\s*(\d+)\s*//', content)
    runs = int(runs_match.group(1)) if runs_match else None
    
    # 查找 numRuns := 10
    num_runs_match = re.search(r'numRuns\s*:=\s*(\d+)', content)
    num_runs = int(num_runs_match.group(1)) if num_runs_match else None
    
    # 查找 inferencesPerRun := 200
    inferences_match = re.search(r'inferencesPerRun\s*:=\s*(\d+)', content)
    inferences_per_run = int(inferences_match.group(1)) if inferences_match else None
    
    # 计算总推理次数
    if num_runs and inferences_per_run:
        total = num_runs * inferences_per_run
    elif runs:
        total = runs
    else:
        total = None
    
    return {
        'runs': runs,
        'num_runs': num_runs,
        'inferences_per_run': inferences_per_run,
        'total': total
    }

def count_inferences_in_python_file(filepath):
    """从Python文件中提取推理次数配置"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找 runs = 200 # 每轮200次推理
    runs_match = re.search(r'runs\s*=\s*(\d+)', content)
    runs = int(runs_match.group(1)) if runs_match else None
    
    # 查找 num_runs = 10
    num_runs_match = re.search(r'num_runs\s*=\s*(\d+)', content)
    num_runs = int(num_runs_match.group(1)) if num_runs_match else None
    
    # 查找 inferences_per_run = 200
    inferences_match = re.search(r'inferences_per_run\s*=\s*(\d+)', content)
    inferences_per_run = int(inferences_match.group(1)) if inferences_match else None
    
    # 计算总推理次数
    if num_runs and inferences_per_run:
        total = num_runs * inferences_per_run
    elif runs:
        total = runs
    else:
        total = None
    
    return {
        'runs': runs,
        'num_runs': num_runs,
        'inferences_per_run': inferences_per_run,
        'total': total
    }

def main():
    test_dir = Path('d:/mlz/trae_projects/1/yolo-go-detector/test')
    
    # 收集所有测试程序
    test_programs = []
    
    # Go测试程序
    go_files = list(test_dir.glob('benchmark/*.go'))
    for f in go_files:
        if f.name not in ['test_env.go']:
            config = count_inferences_in_go_file(f)
            test_programs.append({
                'name': f.name,
                'language': 'Go',
                **config
            })
    
    # Python测试程序
    python_files = list(test_dir.glob('python/*.py'))
    for f in python_files:
        if f.name not in ['data_analysis.py']:
            config = count_inferences_in_python_file(f)
            test_programs.append({
                'name': f.name,
                'language': 'Python',
                **config
            })
    
    # 打印统计结果
    print("=" * 90)
    print("测试程序推理次数统计 (改进版)")
    print("=" * 90)
    print(f"\n总计测试程序数量: {len(test_programs)}")
    print("\n详细列表:")
    print("-" * 90)
    print(f"{'程序名称':<50} {'语言':<8} {'轮数':<8} {'每轮':<8} {'总计':<12}")
    print("-" * 90)
    
    total_inferences = 0
    known_count = 0
    
    # 按语言排序
    test_programs.sort(key=lambda x: (x['language'], x['name']))
    
    for prog in test_programs:
        num_runs_str = str(prog['num_runs']) if prog['num_runs'] else '-'
        inferences_str = str(prog['inferences_per_run']) if prog['inferences_per_run'] else '-'
        total_str = f"{prog['total']:,}" if prog['total'] else '未知'
        
        print(f"{prog['name']:<50} {prog['language']:<8} {num_runs_str:<8} {inferences_str:<8} {total_str:<12}")
        
        if prog['total']:
            total_inferences += prog['total']
            known_count += 1
    
    print("-" * 90)
    print(f"\n已知推理次数的程序: {known_count}/{len(test_programs)}")
    print(f"已知程序总推理次数: {total_inferences:,} 次")
    
    # 估算未知程序
    unknown_count = len(test_programs) - known_count
    if unknown_count > 0:
        # 根据已知程序的平均值估算
        if known_count > 0:
            avg_known = total_inferences / known_count
            estimated_total = total_inferences + (unknown_count * avg_known)
            print(f"\n估算说明:")
            print(f"  - 已知程序平均推理次数: {avg_known:,.0f} 次")
            print(f"  - 未知程序数量: {unknown_count} 个")
            print(f"  - 估算总推理次数: {estimated_total:,.0f} 次")
    
    print("\n" + "=" * 90)
    
    # 分类统计
    print("\n按类别统计:")
    print("-" * 90)
    
    categories = {
        'reinforced': {'name': '强化测试', 'count': 0, 'total': 0},
        'baseline': {'name': '基准测试', 'count': 0, 'total': 0},
        'cold_start': {'name': '冷启动测试', 'count': 0, 'total': 0},
        'thread': {'name': '线程配置测试', 'count': 0, 'total': 0},
        'architecture': {'name': '架构对比测试', 'count': 0, 'total': 0},
        'memory': {'name': '内存测试', 'count': 0, 'total': 0},
        'other': {'name': '其他测试', 'count': 0, 'total': 0},
    }
    
    for prog in test_programs:
        name_lower = prog['name'].lower()
        if 'reinforced' in name_lower:
            cat = 'reinforced'
        elif 'baseline' in name_lower:
            cat = 'baseline'
        elif 'cold_start' in name_lower:
            cat = 'cold_start'
        elif 'thread' in name_lower:
            cat = 'thread'
        elif 'architecture' in name_lower:
            cat = 'architecture'
        elif 'memory' in name_lower:
            cat = 'memory'
        else:
            cat = 'other'
        
        categories[cat]['count'] += 1
        if prog['total']:
            categories[cat]['total'] += prog['total']
    
    for cat_key, cat_info in categories.items():
        if cat_info['count'] > 0:
            avg_str = f"{cat_info['total'] / cat_info['count']:,.0f}" if cat_info['total'] > 0 else "未知"
            print(f"  {cat_info['name']}: {cat_info['count']} 个程序, 总推理 {cat_info['total']:,} 次 (平均 {avg_str} 次)")
    
    print("\n" + "=" * 90)

if __name__ == '__main__':
    main()
