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

print("===== Python 冷启动时间对比分析测试 ====")
print(f"模型路径: {model_path}")

# 执行3次独立测试
test_count = 3
all_cold_start_times = []
all_avg_stable_latencies = []
all_min_stable_latencies = []
all_max_stable_latencies = []
all_p50_stable_latencies = []
all_p90_stable_latencies = []
all_p99_stable_latencies = []
all_start_rss = []
all_cold_start_rss = []
all_stable_rss = []

for test_idx in range(1, test_count + 1):
    print(f"\n=== 独立测试 {test_idx}/{test_count} ===")

    # 创建 Session
    print("创建 InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        
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

    # 使用与 Go 完全一致的输入数据（从文件加载）
    print("加载输入数据...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"输入数据加载成功: {input_data_path}")
    except Exception as e:
        print(f"加载输入数据失败: {e}")
        sys.exit(1)

    # 内存采样点 1：Session 创建后（Start RSS）
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    print(f"Start RSS: {start_rss:.2f} MB")

    # 测试冷启动时间
    print("\n===== 测试冷启动时间 =====")
    t0 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t1 = time.perf_counter()
    cold_start_time = (t1 - t0) * 1000.0
    print(f"冷启动时间: {cold_start_time:.3f} ms")

    # 内存采样点 2：冷启动后（Cold Start RSS）
    cold_start_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    print(f"Cold Start RSS: {cold_start_rss:.2f} MB")

    # 预热阶段
    print("\n===== 预热阶段 =====")
    warmup_count = 10
    warmup_latencies = []
    for i in range(warmup_count):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        warmup_latencies.append(dt)

    # 稳定状态测试
    print("\n===== 稳定状态测试 =====")
    stable_count = 100
    stable_latencies = []
    peak_rss = cold_start_rss

    for i in range(stable_count):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        stable_latencies.append(dt)

        # 每10次推理采样一次内存，记录峰值
        if i % 10 == 0:
            current_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
            if current_rss > peak_rss:
                peak_rss = current_rss

    # 内存采样点 3：稳定状态后（Stable RSS）
    stable_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    print(f"\nStable RSS: {stable_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")

    # 计算稳定状态的统计数据
    avg_stable_latency = sum(stable_latencies) / len(stable_latencies)
    min_stable_latency = min(stable_latencies)
    max_stable_latency = max(stable_latencies)
    p50_stable_latency = np.percentile(stable_latencies, 50)
    p90_stable_latency = np.percentile(stable_latencies, 90)
    p99_stable_latency = np.percentile(stable_latencies, 99)

    # 保存本次测试结果
    all_cold_start_times.append(cold_start_time)
    all_avg_stable_latencies.append(avg_stable_latency)
    all_min_stable_latencies.append(min_stable_latency)
    all_max_stable_latencies.append(max_stable_latency)
    all_p50_stable_latencies.append(p50_stable_latency)
    all_p90_stable_latencies.append(p90_stable_latency)
    all_p99_stable_latencies.append(p99_stable_latency)
    all_start_rss.append(start_rss)
    all_cold_start_rss.append(cold_start_rss)
    all_stable_rss.append(stable_rss)

    print(f"测试 {test_idx} 完成: 冷启动时间={cold_start_time:.3f} ms, 稳定状态平均时间={avg_stable_latency:.3f} ms")

# 计算3次测试的平均值
cold_start_time = np.mean(all_cold_start_times)
avg_stable_latency = np.mean(all_avg_stable_latencies)
min_stable_latency = np.mean(all_min_stable_latencies)
max_stable_latency = np.mean(all_max_stable_latencies)
p50_stable_latency = np.mean(all_p50_stable_latencies)
p90_stable_latency = np.mean(all_p90_stable_latencies)
p99_stable_latency = np.mean(all_p99_stable_latencies)
start_rss = np.mean(all_start_rss)
cold_start_rss = np.mean(all_cold_start_rss)
stable_rss = np.mean(all_stable_rss)

# 计算标准差
std_dev_stable = np.std(all_avg_stable_latencies)
# 计算变异系数
coeff_var_stable = (std_dev_stable / avg_stable_latency) * 100
# 计算FPS
fps = 1000.0 / avg_stable_latency

# 输出结果
print("\n===== 冷启动与稳定状态对比结果 =====")
print(f"冷启动时间: {cold_start_time:.3f} ms")
print(f"稳定状态平均时间: {avg_stable_latency:.3f} ms")
print(f"冷启动时间 / 稳定状态平均时间: {cold_start_time/avg_stable_latency:.2f} 倍")
print("\n===== 稳定状态详细统计 =====")
print(f"平均延迟: {avg_stable_latency:.3f} ms")
print(f"标准差: {std_dev_stable:.3f} ms")
print(f"变异系数: {coeff_var_stable:.2f}%")
print(f"FPS: {fps:.2f}")
print(f"最小延迟: {min_stable_latency:.3f} ms")
print(f"最大延迟: {max_stable_latency:.3f} ms")
print(f"P50延迟: {p50_stable_latency:.3f} ms")
print(f"P90延迟: {p90_stable_latency:.3f} ms")
print(f"P99延迟: {p99_stable_latency:.3f} ms")
print("\n===== 内存使用情况 =====")
print(f"Start RSS: {start_rss:.2f} MB")
print(f"Cold Start RSS: {cold_start_rss:.2f} MB")
print(f"Stable RSS: {stable_rss:.2f} MB")
print(f"内存增长 (Start -> Cold Start): {cold_start_rss-start_rss:.2f} MB")
print(f"内存增长 (Cold Start -> Stable): {stable_rss-cold_start_rss:.2f} MB")

# 保存结果
result_path = os.path.join(current_dir, '..', '..', 'results', 'python_cold_start_result.txt')
print(f"\n保存结果到: {result_path}")

# 构建结果字符串
result_lines = [
    "===== Python 冷启动时间对比分析测试结果 =====",
    f"冷启动时间: {cold_start_time:.3f} ms",
    f"稳定状态平均时间: {avg_stable_latency:.3f} ms",
    f"冷启动时间 / 稳定状态平均时间: {cold_start_time/avg_stable_latency:.2f} 倍",
    "",
    "===== 稳定状态详细统计 =====",
    f"平均延迟: {avg_stable_latency:.3f} ms",
    f"标准差: {std_dev_stable:.3f} ms",
    f"变异系数: {coeff_var_stable:.2f}%",
    f"FPS: {fps:.2f}",
    f"最小延迟: {min_stable_latency:.3f} ms",
    f"最大延迟: {max_stable_latency:.3f} ms",
    f"P50延迟: {p50_stable_latency:.3f} ms",
    f"P90延迟: {p90_stable_latency:.3f} ms",
    f"P99延迟: {p99_stable_latency:.3f} ms",
    "",
    "===== 内存使用情况 =====",
    f"Start RSS: {start_rss:.2f} MB",
    f"Cold Start RSS: {cold_start_rss:.2f} MB",
    f"Stable RSS: {stable_rss:.2f} MB",
    f"内存增长 (Start -> Cold Start): {cold_start_rss-start_rss:.2f} MB",
    f"内存增长 (Cold Start -> Stable): {stable_rss-cold_start_rss:.2f} MB"
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
    
    # 显示完整文件内容
    print("\n文件完整内容:")
    print(content)
    
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
        
        # 显示完整文件内容
        print("\n文件完整内容:")
        print(content)
        
    except Exception as e2:
        print(f"GBK编码写入失败: {e2}")
        
        # 方法3: 使用二进制模式写入
        try:
            print("尝试使用二进制模式写入...")
            with open(result_path, 'wb') as f:
                for line in result_lines:
                    f.write((line + '\n').encode('utf-8'))
            print("二进制模式写入成功!")
            
            # 验证文件内容
            with open(result_path, 'rb') as f:
                content = f.read().decode('utf-8')
            print(f"文件前500字符: {content[:500]}...")
            
            # 显示完整文件内容
            print("\n文件完整内容:")
            print(content)
            
        except Exception as e3:
            print(f"二进制模式写入失败: {e3}")

print(f"\n结果已保存到: {result_path}")
print("\n===== 冷启动时间对比分析测试完成 ====")
