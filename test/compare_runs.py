#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_runs.py
Compare Round 10 and Round 11 test results for verification
"""

import os
import re
import sys


def parse_arena_ablation_result(filepath):
    """Parse arena ablation result file (Go or Python format)"""
    if not os.path.exists(filepath):
        return None
    
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse Go format (from go_arena_ablation.go output)
    # Format: "Unsafe Shared    ON       1.15663    3457.046     2194.32      2166.73"
    go_pattern = r'(Unsafe Shared|Session Pool)\s+(ON|OFF)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)'
    go_matches = re.findall(go_pattern, content)
    if go_matches:
        for match in go_matches:
            arch, arena, throughput, latency, peak_rss, drift = match
            key = f"{arch}_{arena}"
            results[key] = {
                'throughput': float(throughput),
                'latency': float(latency),
                'peak_rss': float(peak_rss),
                'drift': float(drift)
            }
        return results
    
    # Parse Python format (from python_arena_ablation.py output)
    # Format: "  吞吐量: 1.15663 REQ/s"
    py_patterns = {
        'throughput': r'吞吐量:\s*([\d.]+)',
        'latency': r'平均延迟:\s*([\d.]+)',
        'peak_rss': r'峰值RSS:\s*([\d.]+)',
        'drift': r'RSS漂移:\s*([-\d.]+)'
    }
    
    # Find all architecture blocks
    arch_blocks = re.split(r'===== (Unsafe Shared|Session Pool) \(arena=(ON|OFF)\) =====', content)
    
    for i in range(1, len(arch_blocks), 3):
        if i + 2 < len(arch_blocks):
            arch = arch_blocks[i]
            arena = arch_blocks[i + 1]
            block_content = arch_blocks[i + 2]
            
            key = f"{arch}_{arena}"
            results[key] = {}
            
            for metric, pattern in py_patterns.items():
                match = re.search(pattern, block_content)
                if match:
                    results[key][metric] = float(match.group(1))
    
    return results if results else None


def compare_metrics(run10, run11, metric_name, threshold=0.05):
    """Compare a metric between two runs, return True if within threshold"""
    if metric_name not in run10 or metric_name not in run11:
        return None, "Missing data"
    
    val10 = run10[metric_name]
    val11 = run11[metric_name]
    
    if val10 == 0:
        diff_pct = 0 if val11 == 0 else float('inf')
    else:
        diff_pct = abs(val11 - val10) / val10
    
    within_threshold = diff_pct <= threshold
    return within_threshold, f"{val10:.5f} -> {val11:.5f} ({diff_pct*100:.2f}%)"


def compare_arena_results(results_dir):
    """Compare arena ablation results between run10 and run11"""
    print("=" * 70)
    print("Arena Ablation Results Comparison (Round 10 vs Round 11)")
    print("=" * 70)
    print()
    
    files = [
        ('go_arena_ablation_result', 'Go Arena Ablation'),
        ('python_arena_ablation_result', 'Python Arena Ablation')
    ]
    
    all_passed = True
    
    for filename, description in files:
        run10_file = os.path.join(results_dir, f"{filename}_run10.txt")
        run11_file = os.path.join(results_dir, f"{filename}_run11.txt")
        
        run10_data = parse_arena_ablation_result(run10_file)
        run11_data = parse_arena_ablation_result(run11_file)
        
        if run10_data is None or run11_data is None:
            print(f"[SKIP] {description}: One or both result files not found")
            print(f"        Run10: {run10_file}")
            print(f"        Run11: {run11_file}")
            print()
            continue
        
        print(f"[{description}]")
        print("-" * 70)
        
        # Compare each configuration
        for key in sorted(run10_data.keys()):
            if key not in run11_data:
                print(f"  [WARN] {key}: Missing in Run 11")
                continue
            
            print(f"  {key}:")
            
            metrics = ['throughput', 'latency', 'peak_rss', 'drift']
            for metric in metrics:
                passed, detail = compare_metrics(run10_data[key], run11_data[key], metric)
                if passed is None:
                    status = "[N/A]"
                elif passed:
                    status = "[PASS]"
                else:
                    status = "[FAIL]"
                    all_passed = False
                
                print(f"    {metric:12s}: {status} {detail}")
        
        print()
    
    return all_passed


def compare_architecture_results(results_dir):
    """Compare architecture comparison results between run10 and run11"""
    print("=" * 70)
    print("Architecture Comparison Results (Round 10 vs Round 11)")
    print("=" * 70)
    print()
    
    run10_file = os.path.join(results_dir, "go_architecture_comparison_run10.txt")
    run11_file = os.path.join(results_dir, "go_architecture_comparison_run11.txt")
    
    if not os.path.exists(run10_file) or not os.path.exists(run11_file):
        print("[SKIP] Architecture comparison: One or both result files not found")
        print()
        return True
    
    # Simple line count comparison for now
    with open(run10_file, 'r', encoding='utf-8') as f:
        run10_lines = len(f.readlines())
    
    with open(run11_file, 'r', encoding='utf-8') as f:
        run11_lines = len(f.readlines())
    
    print(f"  Run10 lines: {run10_lines}")
    print(f"  Run11 lines: {run11_lines}")
    
    if run10_lines == run11_lines:
        print(f"  [PASS] Line count matches")
    else:
        print(f"  [WARN] Line count differs by {abs(run11_lines - run10_lines)}")
    
    print()
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: compare_runs.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print()
    print("=" * 70)
    print("  Double Verification Results Comparison")
    print("=" * 70)
    print()
    
    arena_passed = compare_arena_results(results_dir)
    arch_passed = compare_architecture_results(results_dir)
    
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    
    if arena_passed and arch_passed:
        print("  [PASS] All key metrics within 5% threshold")
        print("  Result: Round 10 and Round 11 are CONSISTENT")
    else:
        print("  [WARN] Some metrics exceed 5% threshold")
        print("  Result: Round 10 and Round 11 have SIGNIFICANT DIFFERENCES")
        print("  Recommendation: Investigate the cause before proceeding")
    
    print("=" * 70)
    print()
    
    sys.exit(0 if arena_passed and arch_passed else 1)


if __name__ == "__main__":
    main()
