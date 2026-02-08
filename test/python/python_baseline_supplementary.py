import onnxruntime as ort
import numpy as np
import time
import os
import psutil

def get_process_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def calculate_metrics(latencies):
    if len(latencies) == 0:
        return {}
    
    latencies_array = np.array(latencies)
    return {
        'avg': np.mean(latencies_array),
        'p50': np.percentile(latencies_array, 50),
        'p90': np.percentile(latencies_array, 90),
        'p99': np.percentile(latencies_array, 99),
        'min': np.min(latencies_array),
        'max': np.max(latencies_array)
    }

def run_baseline_test(model_path, num_threads):
    print(f"===== 实验编号 S-B{num_threads}: intra_op_num_threads={num_threads} =====")
    print("执行路径：Baseline InferenceSession（不启用 io_binding，不预分配输出）")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print("创建 InferenceSession...")
    try:
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        print("InferenceSession 创建成功!")
    except Exception as e:
        print(f"错误: 创建 InferenceSession 失败: {e}")
        return None, None
    
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    
    print(f"输入名称: {input_name}")
    print(f"输入形状: {input_shape}")
    
    print("生成固定随机输入数据...")
    np.random.seed(42)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    start_rss = get_process_rss()
    print(f"Start RSS: {start_rss:.2f} MB")
    
    print("Warming up...")
    for i in range(10):
        outputs = sess.run(None, {input_name: input_data})
    
    warmup_rss = get_process_rss()
    print(f"Warmup 后 RSS: {warmup_rss:.2f} MB")
    
    print("开始基准测试...")
    latencies = []
    for i in range(100):
        start = time.time()
        outputs = sess.run(None, {input_name: input_data})
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)
    
    peak_rss = get_process_rss()
    print(f"Peak RSS: {peak_rss:.2f} MB")
    
    metrics = calculate_metrics(latencies)
    
    print(f"性能指标: avg={metrics['avg']:.2f} ms, p50={metrics['p50']:.2f} ms, "
          f"p90={metrics['p90']:.2f} ms, p99={metrics['p99']:.2f} ms, "
          f"min={metrics['min']:.2f} ms, max={metrics['max']:.2f} ms")
    
    engineering_metrics = {
        'tensor_allocation_count': 'N/A (baseline)',
        'io_binding_enabled': False,
        'session_creation_count': 1,
        'peak_rss': peak_rss
    }
    
    print(f"工程指标: Tensor分配次数={engineering_metrics['tensor_allocation_count']}, "
          f"I/O Binding={engineering_metrics['io_binding_enabled']}, "
          f"Session创建次数={engineering_metrics['session_creation_count']}, "
          f"峰值RSS={engineering_metrics['peak_rss']:.2f} MB")
    
    return metrics, engineering_metrics

def main():
    print("===== Python Baseline 补充实验 =====")
    print("实验性质：工程级接口能力评估对照（非语言级性能比较）")
    print("对照策略：Python 仍使用 baseline（不启用 io_binding，不预分配输出）")
    print("原因：Python 侧 io_binding 的行为高度依赖版本与绑定方式，")
    print("难以在工程层面保证与 Go 完全一致，因此未纳入补充实验对照。")
    print()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, "third_party", "yolo11x.onnx")
    
    print(f"当前目录: {current_dir}")
    print(f"项目根路径: {project_root}")
    print(f"模型路径: {model_path}")
    print()
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    thread_configs = [1, 2, 4, 8]
    results = {}
    engineering_results = {}
    
    for i, num_threads in enumerate(thread_configs):
        perf_metrics, eng_metrics = run_baseline_test(model_path, num_threads)
        if perf_metrics is not None:
            results[num_threads] = perf_metrics
            engineering_results[num_threads] = eng_metrics
        print()
    
    save_results(results, engineering_results)
    print("===== 补充实验完成 =====")

def save_results(results, engineering_results):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    result_path = os.path.join(project_root, "results", "python_baseline_supplementary.txt")
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Baseline 补充实验结果 =====\n")
        f.write("实验性质：工程级接口能力评估对照（非语言级性能比较）\n")
        f.write("执行路径：Baseline InferenceSession（不启用 io_binding，不预分配输出）\n")
        f.write("对照策略：Python 仍使用 baseline（不启用 io_binding）\n")
        f.write("原因：Python 侧 io_binding 的行为高度依赖版本与绑定方式，\n")
        f.write("难以在工程层面保证与 Go 完全一致，因此未纳入补充实验对照。\n\n")
        
        f.write("性能指标：\n")
        f.write("线程配置\t平均延迟\tP50\tP90\tP99\t最小值\t最大值\n")
        for num_threads in [1, 2, 4, 8]:
            if num_threads in results:
                metrics = results[num_threads]
                f.write(f"{num_threads}\t{metrics['avg']:.2f}\t{metrics['p50']:.2f}\t"
                       f"{metrics['p90']:.2f}\t{metrics['p99']:.2f}\t"
                       f"{metrics['min']:.2f}\t{metrics['max']:.2f}\n")
        
        f.write("\n工程指标：\n")
        f.write("线程配置\tTensor分配次数\tI/O Binding\tSession创建次数\t峰值RSS(MB)\n")
        for num_threads in [1, 2, 4, 8]:
            if num_threads in engineering_results:
                metrics = engineering_results[num_threads]
                f.write(f"{num_threads}\t{metrics['tensor_allocation_count']}\t"
                       f"{metrics['io_binding_enabled']}\t{metrics['session_creation_count']}\t"
                       f"{metrics['peak_rss']:.2f}\n")
        
        f.write("\n不可比声明：\n")
        f.write("本节实验通过 AdvancedSession 与 I/O Binding 引入了工程级执行路径优化，\n")
        f.write("其内存分配和执行调度机制与前文 baseline 测试存在本质差异，\n")
        f.write("因此结果不用于修正语言级性能结论，仅用于评估 Go 在 ONNX 推理任务中的工程接口性能潜力。\n")
    
    print(f"结果已保存到: {result_path}")

if __name__ == "__main__":
    main()
