# data_analysis.py
# 数据分析脚本
# 
# 测试目的：
# - 执行独立样本t-test，比较Python和Go的性能差异
# - 计算95%置信区间
# - 生成详细的分析报告
# - 确保统计显著性

import numpy as np
import scipy.stats as stats
import os
import sys
import re
from dataclasses import dataclass

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

@dataclass
class PerformanceData:
    python_times: list
    go_times: list
    model_name: str

@dataclass
class TestResult:
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    mean_diff: float
    significant: bool
    model_name: str

def load_latency_data(file_path):
    """从文件加载延迟数据"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return []
    
    times = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('====='):
                try:
                    time_val = float(line)
                    times.append(time_val)
                except ValueError:
                    pass
    
    return times

def load_reinforced_data(model_name):
    """加载强化测试数据"""
    # 加载Python数据
    python_file = os.path.join(base_path, "results", f"python_reinforced{('_small' if model_name == 'yolo11n' else '')}_latency_data.txt")
    python_times = load_latency_data(python_file)
    
    # 加载Go数据
    go_file = os.path.join(base_path, "results", f"go_reinforced{('_small' if model_name == 'yolo11n' else '')}_latency_data.txt")
    go_times = load_latency_data(go_file)
    
    return PerformanceData(
        python_times=python_times,
        go_times=go_times,
        model_name=model_name
    )

def perform_ttest(data):
    """执行独立样本t-test"""
    if len(data.python_times) == 0 or len(data.go_times) == 0:
        print(f"警告: 缺少{data.model_name}的测试数据")
        return None
    
    # 计算均值
    python_mean = np.mean(data.python_times)
    go_mean = np.mean(data.go_times)
    mean_diff = python_mean - go_mean
    
    # 执行t-test
    t_stat, p_value = stats.ttest_ind(data.python_times, data.go_times, equal_var=False)
    
    # 计算95%置信区间
    n1 = len(data.python_times)
    n2 = len(data.go_times)
    df = n1 + n2 - 2
    std1 = np.std(data.python_times, ddof=1)
    std2 = np.std(data.go_times, ddof=1)
    
    # 标准误
    se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    # 95%置信区间
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_critical * se
    ci_upper = mean_diff + t_critical * se
    
    # 判断显著性
    significant = p_value < 0.05
    
    return TestResult(
        t_statistic=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_diff=mean_diff,
        significant=significant,
        model_name=data.model_name
    )

def analyze_cold_start_data():
    """分析冷启动数据"""
    print("\n===== 冷启动数据分析 =====")
    
    # 加载Python冷启动数据
    python_file = os.path.join(base_path, "results", "python_cold_start_detailed_log.txt")
    go_file = os.path.join(base_path, "results", "go_cold_start_decomposition_log.txt")
    
    if not os.path.exists(python_file) or not os.path.exists(go_file):
        print("警告: 冷启动数据文件不存在")
        return
    
    # 解析Python冷启动数据
    python_data = parse_cold_start_log(python_file)
    go_data = parse_cold_start_log(go_file)
    
    # 比较冷启动时间
    for model in ["yolo11x", "yolo11n"]:
        if model in python_data and model in go_data:
            python_times = python_data[model]
            go_times = go_data[model]
            
            print(f"\n模型: {model}")
            print(f"Python 平均冷启动时间: {np.mean(python_times):.5f} ms")
            print(f"Go 平均冷启动时间: {np.mean(go_times):.5f} ms")
            print(f"差异: {np.mean(python_times) - np.mean(go_times):.5f} ms")
            
            # 执行t-test
            if len(python_times) > 1 and len(go_times) > 1:
                t_stat, p_value = stats.ttest_ind(python_times, go_times, equal_var=False)
                print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                print(f"显著性: {'显著' if p_value < 0.05 else '不显著'}")

def parse_cold_start_log(file_path):
    """解析冷启动日志文件"""
    data = {}
    current_model = None
    times = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测模型切换
            if "大模型 (YOLO11x)" in line:
                if current_model and times:
                    data[current_model] = times
                current_model = "yolo11x"
                times = []
            elif "轻模型 (YOLO11n)" in line:
                if current_model and times:
                    data[current_model] = times
                current_model = "yolo11n"
                times = []
            
            # 提取总冷启动时间
            elif "总冷启动时间:" in line:
                match = re.search(r'总冷启动时间: ([\d.]+) ms', line)
                if match:
                    time_val = float(match.group(1))
                    times.append(time_val)
    
    # 保存最后一个模型的数据
    if current_model and times:
        data[current_model] = times
    
    return data

def analyze_memory_data():
    """分析内存数据"""
    print("\n===== 内存使用数据分析 =====")
    
    # 加载Python内存数据
    python_file = os.path.join(base_path, "results", "python_memory_detailed_log.txt")
    go_file = os.path.join(base_path, "results", "go_memory_detailed_log.txt")
    
    if not os.path.exists(python_file) or not os.path.exists(go_file):
        print("警告: 内存数据文件不存在")
        return
    
    # 解析内存数据
    python_data = parse_memory_log(python_file)
    go_data = parse_memory_log(go_file)
    
    # 比较内存使用
    for model in ["yolo11x", "yolo11n"]:
        if model in python_data and model in go_data:
            python_mem = python_data[model]
            go_mem = go_data[model]
            
            print(f"\n模型: {model}")
            print(f"Python 平均内存增加: {np.mean(python_mem):.5f} MB")
            print(f"Go 平均内存增加: {np.mean(go_mem):.5f} MB")
            print(f"差异: {np.mean(python_mem) - np.mean(go_mem):.5f} MB")

def parse_memory_log(file_path):
    """解析内存日志文件"""
    data = {}
    current_model = None
    memory_increases = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测模型切换
            if "大模型 (YOLO11x)" in line:
                if current_model and memory_increases:
                    data[current_model] = memory_increases
                current_model = "yolo11x"
                memory_increases = []
            elif "轻模型 (YOLO11n)" in line:
                if current_model and memory_increases:
                    data[current_model] = memory_increases
                current_model = "yolo11n"
                memory_increases = []
            
            # 提取内存增加量
            elif "内存增加量:" in line:
                match = re.search(r'内存增加量: ([\d.]+) MB', line)
                if match:
                    mem_val = float(match.group(1))
                    memory_increases.append(mem_val)
    
    # 保存最后一个模型的数据
    if current_model and memory_increases:
        data[current_model] = memory_increases
    
    return data

def generate_report(results):
    """生成分析报告"""
    report_path = os.path.join(base_path, "results", "analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== 性能分析报告 =====\n\n")
        
        # 性能比较
        f.write("===== 性能比较 =====\n\n")
        for result in results:
            f.write(f"模型: {result.model_name}\n")
            f.write(f"t-statistic: {result.t_statistic:.4f}\n")
            f.write(f"p-value: {result.p_value:.4f}\n")
            f.write(f"95% 置信区间: [{result.ci_lower:.4f}, {result.ci_upper:.4f}] ms\n")
            f.write(f"平均差异: {result.mean_diff:.4f} ms\n")
            f.write(f"显著性: {'显著' if result.significant else '不显著'}\n\n")
        
        # 结论
        f.write("===== 结论 =====\n\n")
        for result in results:
            if result.significant:
                if result.mean_diff > 0:
                    f.write(f"对于{result.model_name}模型，Go的性能显著优于Python (p < 0.05)\n")
                else:
                    f.write(f"对于{result.model_name}模型，Python的性能显著优于Go (p < 0.05)\n")
            else:
                f.write(f"对于{result.model_name}模型，Python和Go的性能差异不显著 (p >= 0.05)\n")
    
    print(f"分析报告已保存到: {report_path}")

def main():
    print("===== 数据分析 =====")
    
    # 加载数据
    print("加载测试数据...")
    data_large = load_reinforced_data("yolo11x")
    data_small = load_reinforced_data("yolo11n")
    
    # 执行t-test
    print("执行t-test分析...")
    results = []
    
    result_large = perform_ttest(data_large)
    if result_large:
        results.append(result_large)
        print(f"\n大模型 (YOLO11x):")
        print(f"t-statistic: {result_large.t_statistic:.4f}")
        print(f"p-value: {result_large.p_value:.4f}")
        print(f"95% 置信区间: [{result_large.ci_lower:.4f}, {result_large.ci_upper:.4f}] ms")
        print(f"平均差异: {result_large.mean_diff:.4f} ms")
        print(f"显著性: {'显著' if result_large.significant else '不显著'}")
    
    result_small = perform_ttest(data_small)
    if result_small:
        results.append(result_small)
        print(f"\n轻模型 (YOLO11n):")
        print(f"t-statistic: {result_small.t_statistic:.4f}")
        print(f"p-value: {result_small.p_value:.4f}")
        print(f"95% 置信区间: [{result_small.ci_lower:.4f}, {result_small.ci_upper:.4f}] ms")
        print(f"平均差异: {result_small.mean_diff:.4f} ms")
        print(f"显著性: {'显著' if result_small.significant else '不显著'}")
    
    # 分析冷启动数据
    analyze_cold_start_data()
    
    # 分析内存数据
    analyze_memory_data()
    
    # 生成报告
    if results:
        generate_report(results)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()
