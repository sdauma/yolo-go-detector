#!/usr/bin/env python3
"""
统计41个测试程序的实际推理次数

⚠ 已过时：此脚本仅能解析约 9/41 个程序的推理次数（不含预热warmup）。
准确的手工逐文件计数见：测试程序与图表生成程序完整清单.md §9。
保留此文件仅供历史参考，请勿用于正式统计。
"""

import os
import re
from pathlib import Path

def count_inferences_in_file(filepath):
    """从文件中提取推理次数配置"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找 runs/numRuns 配置
    runs_patterns = [
        r'runs\s*[:=]\s*(\d+)',
        r'numRuns\s*[:=]\s*(\d+)',
        r'num_runs\s*=\s*(\d+)',
    ]
    
    # 查找 numRuns/num_runs 配置（轮数）
    num_runs_patterns = [
        r'numRuns\s*[:=]\s*(\d+)',
        r'num_runs\s*=\s*(\d+)',
    ]
    
    # 查找 inferencesPerRun 配置（每轮推理次数）
    inferences_patterns = [
        r'inferencesPerRun\s*[:=]\s*(\d+)',
        r'inferences_per_run\s*=\s*(\d+)',
    ]
    
    runs = None
    num_runs = None
    inferences_per_run = None
    
    # 查找 runs（总推理次数或每轮推理次数）
    for pattern in runs_patterns:
        match = re.search(pattern, content)
        if match:
            runs = int(match.group(1))
            break
    
    # 查找 numRuns（轮数）
    for pattern in num_runs_patterns:
        match = re.search(pattern, content)
        if match:
            num_runs = int(match.group(1))
            break
    
    # 查找 inferencesPerRun（每轮推理次数）
    for pattern in inferences_patterns:
        match = re.search(pattern, content)
        if match:
            inferences_per_run = int(match.group(1))
            break
    
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
            config = count_inferences_in_file(f)
            test_programs.append({
                'name': f.name,
                'language': 'Go',
                **config
            })
    
    # Python测试程序
    python_files = list(test_dir.glob('python/*.py'))
    for f in python_files:
        if f.name not in ['data_analysis.py']:
            config = count_inferences_in_file(f)
            test_programs.append({
                'name': f.name,
                'language': 'Python',
                **config
            })
    
    # 打印统计结果
    print("=" * 80)
    print("测试程序推理次数统计")
    print("=" * 80)
    print(f"\n总计测试程序数量: {len(test_programs)}")
    print("\n详细列表:")
    print("-" * 80)
    print(f"{'程序名称':<50} {'语言':<8} {'轮数':<8} {'每轮':<8} {'总计':<10}")
    print("-" * 80)
    
    total_inferences = 0
    known_count = 0
    
    for prog in test_programs:
        num_runs_str = str(prog['num_runs']) if prog['num_runs'] else '-'
        inferences_str = str(prog['inferences_per_run']) if prog['inferences_per_run'] else '-'
        total_str = str(prog['total']) if prog['total'] else '未知'
        
        print(f"{prog['name']:<50} {prog['language']:<8} {num_runs_str:<8} {inferences_str:<8} {total_str:<10}")
        
        if prog['total']:
            total_inferences += prog['total']
            known_count += 1
    
    print("-" * 80)
    print(f"\n已知推理次数的程序: {known_count}/{len(test_programs)}")
    print(f"已知程序总推理次数: {total_inferences:,} 次")
    
    # 估算未知程序（假设平均200次）
    unknown_count = len(test_programs) - known_count
    estimated_total = total_inferences + (unknown_count * 200)
    print(f"估算总推理次数: {estimated_total:,} 次 (假设未知程序平均200次)")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
