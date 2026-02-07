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

print("===== Python 线程配置性能测试 ====")
print(f"模型路径: {model_path}")

# 测试的线程配置
thread_configs = [1, 2, 4, 8]

# 对每个线程配置进行测试
for num_threads in thread_configs:
    print(f"\n===== 测试线程配置: intra_op_num_threads={num_threads} ====")
    
    # 执行3次独立测试
    test_count = 3
    all_avg_latencies = []
    all_min_latencies = []
    all_max_latencies = []
    all_p50_latencies = []
    all_p90_latencies = []
    all_p99_latencies = []
    all_start_rss = []
    all_peak_rss = []
    all_stable_rss = []
    
    for test_idx in range(1, test_count + 1):
        print(f"\n=== 独立测试 {test_idx}/{test_count} ===")
        
        # 创建 Session
        print("创建 InferenceSession...")
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = 1
            
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            print("InferenceSession 创建成功!")
        except Exception as e:
            print(f"错误: 创建 InferenceSession 失败: {e}")
            continue
        
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
        
        # 内存采样点 1：Session 创建后、warmup 前（Start RSS）
        process = psutil.Process(os.getpid())
        start_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        print(f"Start RSS: {start_rss:.2f} MB")
        
        # Warmup
        print("Warming up...")
        for i in range(10):
            t0 = time.perf_counter()
            sess.run(None, {input_name: input_data})
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000.0
        
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
            
            # 采样内存，记录峰值
            current_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
            if current_rss > peak_rss:
                peak_rss = current_rss
        
        # 内存采样点 3：Benchmark 后稳定值
        stable_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        print(f"Stable RSS: {stable_rss:.2f} MB")
        print(f"Peak RSS: {peak_rss:.2f} MB")
        
        # 计算结果
        avg_latency = sum(times) / len(times)
        min_latency = min(times)
        max_latency = max(times)
        p50_latency = np.percentile(times, 50)
        p90_latency = np.percentile(times, 90)
        p99_latency = np.percentile(times, 99)
        
        # 保存本次测试结果
        all_avg_latencies.append(avg_latency)
        all_min_latencies.append(min_latency)
        all_max_latencies.append(max_latency)
        all_p50_latencies.append(p50_latency)
        all_p90_latencies.append(p90_latency)
        all_p99_latencies.append(p99_latency)
        all_start_rss.append(start_rss)
        all_peak_rss.append(peak_rss)
        all_stable_rss.append(stable_rss)
        
        print(f"测试 {test_idx} 完成: 平均延迟={avg_latency:.3f} ms")
    
    # 计算3次测试的平均值
    avg_latency = np.mean(all_avg_latencies)
    min_latency = np.mean(all_min_latencies)
    max_latency = np.mean(all_max_latencies)
    p50_latency = np.mean(all_p50_latencies)
    p90_latency = np.mean(all_p90_latencies)
    p99_latency = np.mean(all_p99_latencies)
    start_rss = np.mean(all_start_rss)
    peak_rss = np.mean(all_peak_rss)
    stable_rss = np.mean(all_stable_rss)
    
    # 计算标准差
    std_dev = np.std(all_avg_latencies)
    # 计算变异系数
    coeff_var = (std_dev / avg_latency) * 100
    # 计算FPS
    fps = 1000.0 / avg_latency
    
    print("\n===== 测试结果 =====")
    print(f"平均延迟: {avg_latency:.3f} ms")
    print(f"标准差: {std_dev:.3f} ms")
    print(f"变异系数: {coeff_var:.2f}%")
    print(f"FPS: {fps:.2f}")
    print(f"P50延迟: {p50_latency:.3f} ms")
    print(f"P90延迟: {p90_latency:.3f} ms")
    print(f"P99延迟: {p99_latency:.3f} ms")
    print(f"最小延迟: {min_latency:.3f} ms")
    print(f"最大延迟: {max_latency:.3f} ms")
    print(f"\n===== 内存使用情况 =====")
    print(f"Start RSS: {start_rss:.2f} MB")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    print(f"Stable RSS: {stable_rss:.2f} MB")
    
    # 保存结果
    result_path = os.path.join(current_dir, '..', '..', 'results', f'python_thread_{num_threads}_result.txt')
    print(f"保存结果到: {result_path}")
    
    # 构建结果字符串
    result_lines = [
        f"===== Python 线程配置测试结果 (intra_op_num_threads={num_threads}) =====",
        f"平均延迟: {avg_latency:.3f} ms",
        f"标准差: {std_dev:.3f} ms",
        f"变异系数: {coeff_var:.2f}%",
        f"FPS: {fps:.2f}",
        f"P50延迟: {p50_latency:.3f} ms",
        f"P90延迟: {p90_latency:.3f} ms",
        f"P99延迟: {p99_latency:.3f} ms",
        f"最小延迟: {min_latency:.3f} ms",
        f"最大延迟: {max_latency:.3f} ms",
        "",
        "===== 内存使用情况 =====",
        f"Start RSS: {start_rss:.2f} MB",
        f"Peak RSS: {peak_rss:.2f} MB",
        f"Stable RSS: {stable_rss:.2f} MB"
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

print("\n===== 所有线程配置测试完成 ====")
