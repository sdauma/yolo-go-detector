# python_baseline.py
# Python 基准测试 - Baseline 执行路径
# 
# 重要声明（P0原则）：
# 本测试使用 Python baseline Session 接口（InferenceSession），不启用 I/O Binding。
# 根据 P0 原则，本测试仅用于观察现象，不用于语言级性能结论。
# 
# 测试目的：
# - 观察不同线程配置下的性能趋势
# - 验证 ONNX Runtime 的线程扩展性
# - 不用于语言级线程扩展性结论

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil

# 固定随机种子，确保可复现
np.random.seed(12345)

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"当前目录: {current_dir}")

# 构建模型路径
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
print(f"模型路径: {model_path}")

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
print(f"项目根路径: {base_path}")

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

print("===== Python 基准测试 ====")
print(f"模型路径: {model_path}")

# 创建 Session
print("创建 InferenceSession...")
try:
    sess_options = ort.SessionOptions()
    
    # 显式设置所有 SessionOptions 参数（P2原则：禁止依赖默认值）
    # 线程配置
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    
    # 日志配置（关闭所有日志，避免日志IO干扰性能）
    sess_options.log_severity_level = 3  # 3 = ORT_LOGGING_LEVEL_ERROR
    
    # 性能分析配置（关闭性能分析，避免额外开销）
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # 内存池配置（启用内存池复用）
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 所有未提及的Session参数均使用ONNX Runtime 1.23.2官方默认值
    
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    print("InferenceSession 创建成功!")
except Exception as e:
    print(f"错误: 创建 InferenceSession 失败: {e}")
    sys.exit(1)

# 获取输入信息
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"输入名称: {input_name}")
print(f"输入形状: {input_shape}")

# 使用与 Go 完全一致的输入数据（从文件加载，使用固定种子）
print("加载输入数据...")
input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
try:
    input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
    print(f"输入数据加载成功: {input_data_path}")
except Exception as e:
    print(f"加载输入数据失败: {e}")
    sys.exit(1)

# 内存采样点 1：Session 创建后、warmup 前（Start RSS）
process = psutil.Process(os.getpid())
start_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
print(f"Start RSS: {start_rss:.2f} MB")

# Warmup
print("Warming up...")
for _ in range(10):
	sess.run(None, {input_name: input_data})

# 内存采样点 2：Warmup 后
warmup_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
print(f"Warmup RSS: {warmup_rss:.2f} MB")

# Benchmark
print("Running benchmark...")
runs = 100
times = []
peak_rss = start_rss

for i in range(runs):
	t0 = time.perf_counter()
	sess.run(None, {input_name: input_data})
	t1 = time.perf_counter()
	dt = (t1 - t0) * 1000
	times.append(dt)
	print(f"Run {i+1}: {dt:.3f} ms")

	# 采样内存，记录峰值
	current_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
	if current_rss > peak_rss:
		peak_rss = current_rss

# 内存采样点 3：Benchmark 后稳定值
stable_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
print(f"Stable RSS: {stable_rss:.2f} MB")

# 计算结果
avg_latency = sum(times) / len(times)
min_latency = min(times)
max_latency = max(times)
p50_latency = np.percentile(times, 50)
p90_latency = np.percentile(times, 90)
p99_latency = np.percentile(times, 99)

print("\n===== 测试结果 =====")
print(f"平均延迟: {avg_latency:.3f} ms")
print(f"P50延迟: {p50_latency:.3f} ms")
print(f"P90延迟: {p90_latency:.3f} ms")
print(f"P99延迟: {p99_latency:.3f} ms")
print(f"最小延迟: {min_latency:.3f} ms")
print(f"最大延迟: {max_latency:.3f} ms")
print(f"\n===== 内存使用情况 =====")
print(f"Start RSS: {start_rss:.2f} MB")
print(f"Peak RSS: {peak_rss:.2f} MB")
print(f"Stable RSS: {stable_rss:.2f} MB")
print(f"RSS Drift: {stable_rss - start_rss:.2f} MB")

# 保存结果
result_path = os.path.join(current_dir, '..', '..', 'results', 'python_baseline_result.txt')
print(f"保存结果到: {result_path}")

# 构建结果字符串
result_lines = [
    "===== Python 基准测试结果 =====",
    f"平均延迟: {avg_latency:.3f} ms",
    f"P50延迟: {p50_latency:.3f} ms",
    f"P90延迟: {p90_latency:.3f} ms",
    f"P99延迟: {p99_latency:.3f} ms",
    f"最小延迟: {min_latency:.3f} ms",
    f"最大延迟: {max_latency:.3f} ms",
    "",
    "===== 内存使用情况 =====",
    f"Start RSS: {start_rss:.2f} MB",
    f"Peak RSS: {peak_rss:.2f} MB",
    f"Stable RSS: {stable_rss:.2f} MB",
    f"RSS Drift: {stable_rss - start_rss:.2f} MB"
]

# 尝试多种编码方式
try:
    # 方法1: 使用utf-8编码写入
    print("尝试使用UTF-8编码写入...")
    with open(result_path, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')
    print("UTF-8编码写入成功!")
    
    # 验证文件内容
    print("验证文件内容...")
    with open(result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"文件前500字符: {content[:500]}...")
    
except Exception as e:
    print(f"UTF-8编码写入失败: {e}")
    
    # 方法2: 使用gbk编码写入
    try:
        print("尝试使用GBK编码写入...")
        with open(result_path, 'w', encoding='gbk') as f:
            for line in result_lines:
                f.write(line + '\n')
        print("GBK编码写入成功!")
        
        # 验证文件内容
        with open(result_path, 'r', encoding='gbk') as f:
            content = f.read()
        print(f"文件前500字符: {content[:500]}...")
        
    except Exception as e2:
        print(f"GBK编码写入失败: {e2}")
        
        # 方法3: 使用二进制模式写入
        try:
            print("尝试使用二进制模式写入...")
            with open(result_path, 'wb') as f:
                for line in result_lines:
                    f.write((line + '\n').encode('utf-8'))
            print("二进制模式写入成功!")
            
        except Exception as e3:
            print(f"二进制模式写入失败: {e3}")

print(f"\n结果已保存到: {result_path}")