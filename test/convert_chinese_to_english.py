import os
import re

# 中文到英文的映射
chinese_to_english = {
    # 标题和注释
    "===== Python 推理架构性能对比实验（论文级）=====": "===== Python Inference Architecture Performance Comparison (Paper Level) =====",
    "===== 实验 1: Shared Session 扩展性测试 =====": "===== Experiment 1: Shared Session Scalability Test =====",
    "===== 实验 2: Mutex Protected 串行化测试 =====": "===== Experiment 2: Mutex Protected Serialization Test =====",
    "===== 实验 3: Session Pool 池大小优化测试 =====": "===== Experiment 3: Session Pool Size Optimization Test =====",
    "===== 实验完成 =====": "===== Experiment Completed =====",
    "===== Python 基准测试 =====": "===== Python Baseline Test =====",
    "===== Python 基准测试（10次运行）=====": "===== Python Baseline Test (10 runs) =====",
    "===== 10次测试平均值 =====": "===== Average of 10 runs =====",
    "===== Python Baseline 补充实验 =====": "===== Python Baseline Supplementary Experiment =====",
    "===== 补充实验完成 =====": "===== Supplementary Experiment Completed =====",
    "===== Python 并发推理性能测试（学术版）=====": "===== Python Concurrent Inference Performance Test (Academic Version) =====",
    "===== Session Pool 扩展性测试（并发度 vs CPU 核心数）=====": "===== Session Pool Scalability Test (Concurrency vs CPU Cores) =====",
    "===== 测试完成 =====": "===== Test Completed =====",
    "===== Python CPU 监控测试 =====": "===== Python CPU Monitoring Test =====",
    "===== 测试结果 =====": "===== Test Results =====",
    "===== Python 内存拷贝和线程调度开销测试 =====": "===== Python Memory Copy and Thread Scheduling Overhead Test =====",
    "===== 内存拷贝开销分析 =====": "===== Memory Copy Overhead Analysis =====",
    "===== 线程调度开销测试 =====": "===== Thread Scheduling Overhead Test =====",
    "===== 内存使用分析 =====": "===== Memory Usage Analysis =====",
    "===== Python 输出一致性验证测试 =====": "===== Python Output Consistency Verification Test =====",
    "===== Python Session创建时间测试 =====": "===== Python Session Creation Time Test =====",
    "===== 测试 YOLO11x 模型 =====": "===== Testing YOLO11x Model =====",
    "===== 测试 YOLO11n 模型 =====": "===== Testing YOLO11n Model =====",
    "===== Python Session Pool 消融实验 =====": "===== Python Session Pool Ablation Experiment =====",
    "===== 消融实验汇总 =====": "===== Ablation Experiment Summary =====",
    
    # 通用文本
    "错误: 模型文件不存在": "Error: Model file not found",
    "错误: 创建 InferenceSession 失败": "Error: Failed to create InferenceSession",
    "加载输入数据失败": "Failed to load input data",
    "输入数据加载成功": "Input data loaded successfully",
    "创建 InferenceSession...": "Creating InferenceSession...",
    "InferenceSession 创建成功!": "InferenceSession created successfully!",
    "加载输入数据...": "Loading input data...",
    "Warming up...": "Warming up...",
    "Running benchmark...": "Running benchmark...",
    "开始监测 CPU 使用率...": "Starting CPU usage monitoring...",
    "预热...": "Warming up...",
    "预处理图像...": "Preprocessing image...",
    "执行推理...": "Running inference...",
    "后处理输出...": "Postprocessing output...",
    "检测到": "Detected",
    "个目标": "objects",
    "结果已保存到": "Results saved to",
    "测试完成!": "Test completed!",
    
    # 参数名称
    "并发": "concurrency",
    "池大小": "pool_size",
    "吞吐量": "throughput",
    "平均延迟": "avg_latency",
    "P50": "P50",
    "P90": "P90",
    "P99": "P99",
    "最小延迟": "min_latency",
    "最大延迟": "max_latency",
    "峰值RSS": "peak_rss",
    "RSS漂移": "rss_drift",
    "初始RSS": "start_rss",
    "最终RSS": "end_rss",
    "总请求数": "total_requests",
    "成功请求数": "successful_requests",
    "失败请求数": "failed_requests",
    "总时间": "total_time",
    "CPU使用率": "cpu_usage",
    "平均 CPU 使用率": "avg_cpu",
    "峰值 CPU 使用率": "max_cpu",
    "最低 CPU 使用率": "min_cpu",
    
    # 注释
    "# 控制台输出保留2位小数（便于阅读），文件保存保留5位小数": "# Console output keeps 2 decimal places (for readability), file saves keep 5 decimal places",
    "# 中间数据保留5位小数，符合核心期刊规范": "# Intermediate data keeps 5 decimal places, conforming to core journal standards",
    "# 内存采样点 1：Session 创建后、warmup 前（Start PM）": "# Memory sample point 1: After Session creation, before warmup (Start PM)",
    "# 内存采样点 2：Warmup 后": "# Memory sample point 2: After warmup",
    "# 内存采样点 3：Benchmark 后稳定值": "# Memory sample point 3: Stable value after benchmark",
    "# 采样内存，记录峰值": "# Sample memory, record peak",
    "# 运行10次测试": "# Run 10 tests",
    "# 计算结果": "# Calculate results",
    "# 保存详细日志": "# Save detailed log",
    "# 创建 Session": "# Create Session",
    "# 获取输入信息": "# Get input information",
    "# 使用与 Go 完全一致的输入数据（从文件加载，使用固定种子）": "# Use identical input data as Go (loaded from file with fixed seed)",
    
    # 消融实验特有
    "测试目的：评估池大小和线程配置对Session Pool性能的影响": "Test Purpose: Evaluate the impact of pool size and thread configuration on Session Pool performance",
    "模型": "model",
    "线程": "threads",
    "状态": "status",
    "跳过不合理配置（线程数大于池大小无意义）": "Skip invalid configuration (threads > pool_size is meaningless)",
    "无意义配置": "meaningless configuration",
    "共": "Total",
    "组消融实验（": " ablation experiments (",
    "组完成，": " completed, ",
    "组跳过）": " skipped)",
}

def replace_chinese_with_english(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    replaced = content
    changes = 0
    
    for chinese, english in chinese_to_english.items():
        if chinese in replaced:
            replaced = replaced.replace(chinese, english)
            changes += 1
    
    if changes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(replaced)
        print(f"Updated {file_path}: {changes} replacements")
    else:
        print(f"No changes needed for {file_path}")

def main():
    files = [
        "python_architecture_benchmark.py",
        "python_baseline.py", 
        "python_baseline_supplementary.py",
        "python_concurrent_stress_test_fixed.py",
        "python_cpu_monitoring.py",
        "python_memory_copy_overhead.py",
        "python_output_consistency.py",
        "python_session_creation_benchmark.py",
        "python_session_pool_ablation.py"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.join(script_dir, "python")
    
    for file_name in files:
        file_path = os.path.join(python_dir, file_name)
        if os.path.exists(file_path):
            replace_chinese_with_english(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()