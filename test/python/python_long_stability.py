import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
import csv
from datetime import datetime

# 固定随机种子，确保可复现
np.random.seed(12345)

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"当前目录: {current_dir}")

# 构建模型路径
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
print(f"模型路径: {model_path}")

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

print("===== Python 长时间稳定性测试 =====")
print("测试时长: 10分钟")
print("采样间隔: 1秒")

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
print(f"输入名称: {input_name}")
print(f"输入形状: {input_shape}")

# 使用与 Go 完全一致的随机输入
print("生成随机输入数据...")
input_data = np.random.rand(*input_shape).astype(np.float32)

# 获取进程对象
process = psutil.Process(os.getpid())

# Warmup
print("Warming up...")
for _ in range(10):
    sess.run(None, {input_name: input_data})
print("Warmup 完成!")

# 开始长时间稳定性测试
print("\n===== 开始长时间稳定性测试 =====")
print("测试时长: 10分钟 (600秒)")
print("采样间隔: 1秒")
print("推理模式: 连续推理")

# 测试参数
test_duration = 10 * 60  # 10分钟，单位：秒
sample_interval = 1  # 1秒采样间隔
start_time = time.time()
end_time = start_time + test_duration

# RSS采样数据
rss_samples = []
inference_times = []
peak_rss = 0
min_rss = float('inf')

# 初始RSS采样
initial_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
peak_rss = initial_rss
min_rss = initial_rss
rss_samples.append({
    'timestamp': datetime.now(),
    'elapsed': 0,
    'rss': initial_rss
})
print(f"初始 RSS: {initial_rss:.2f} MB")

# 推理计数器
inference_count = 0

# 主测试循环
while time.time() < end_time:
    # 执行推理
    t0 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000  # 转换为毫秒
    inference_times.append(dt)
    inference_count += 1

    # 采样RSS（每秒采样一次）
    current_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    if current_rss > peak_rss:
        peak_rss = current_rss
    if current_rss < min_rss:
        min_rss = current_rss
    
    elapsed = time.time() - start_time
    rss_samples.append({
        'timestamp': datetime.now(),
        'elapsed': elapsed,
        'rss': current_rss
    })

    # 每分钟输出一次进度
    if inference_count % 60 == 0:
        remaining = end_time - time.time()
        print(f"进度: {inference_count} 次推理, 已运行: {elapsed:.0f}秒, 剩余: {remaining:.0f}秒, 当前RSS: {current_rss:.2f} MB")

    # 等待1秒（确保采样间隔）
    time.sleep(sample_interval)

# 最终RSS采样
final_rss = process.memory_info().rss / 1024 / 1024  # 转换为 MB
rss_samples.append({
    'timestamp': datetime.now(),
    'elapsed': time.time() - start_time,
    'rss': final_rss
})

# 计算统计结果
total_duration = time.time() - start_time
avg_inference_time = np.mean(inference_times)
min_inference_time = np.min(inference_times)
max_inference_time = np.max(inference_times)
p50_inference_time = np.percentile(inference_times, 50)
p90_inference_time = np.percentile(inference_times, 90)
p99_inference_time = np.percentile(inference_times, 99)

# 计算RSS统计
rss_values = [sample['rss'] for sample in rss_samples]
avg_rss = np.mean(rss_values)
rss_drift = final_rss - initial_rss
rss_range = peak_rss - min_rss
rss_range_percent = (rss_range / avg_rss) * 100 if avg_rss > 0 else 0

# 输出测试结果
print(f"\n===== 长时间稳定性测试结果 =====")
print(f"测试时长: {total_duration:.0f}秒")
print(f"推理次数: {inference_count}")
print(f"推理频率: {inference_count / total_duration:.2f} 次/秒")

print(f"\n===== 推理性能统计 =====")
print(f"平均推理时间: {avg_inference_time:.3f} ms")
print(f"P50推理时间: {p50_inference_time:.3f} ms")
print(f"P90推理时间: {p90_inference_time:.3f} ms")
print(f"P99推理时间: {p99_inference_time:.3f} ms")
print(f"最小推理时间: {min_inference_time:.3f} ms")
print(f"最大推理时间: {max_inference_time:.3f} ms")

print(f"\n===== 内存使用统计 =====")
print(f"初始 RSS: {initial_rss:.2f} MB")
print(f"最终 RSS: {final_rss:.2f} MB")
print(f"平均 RSS: {avg_rss:.2f} MB")
print(f"峰值 RSS: {peak_rss:.2f} MB")
print(f"最小 RSS: {min_rss:.2f} MB")
print(f"RSS Drift: {rss_drift:.2f} MB")
print(f"RSS 波动范围: {rss_range:.2f} MB ({rss_range_percent:.2f}%)")

# 保存详细结果
result_path = os.path.join(current_dir, '..', '..', 'results', 'python_long_stability_result.txt')
print(f"\n保存结果到: {result_path}")
try:
    # 使用utf-8编码写入文件
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python 长时间稳定性测试结果 =====\n")
        f.write(f"测试时长: {total_duration:.0f}秒\n")
        f.write(f"推理次数: {inference_count}\n")
        f.write(f"推理频率: {inference_count / total_duration:.2f} 次/秒\n")
        f.write(f"\n===== 推理性能统计 =====\n")
        f.write(f"平均推理时间: {avg_inference_time:.3f} ms\n")
        f.write(f"P50推理时间: {p50_inference_time:.3f} ms\n")
        f.write(f"P90推理时间: {p90_inference_time:.3f} ms\n")
        f.write(f"P99推理时间: {p99_inference_time:.3f} ms\n")
        f.write(f"最小推理时间: {min_inference_time:.3f} ms\n")
        f.write(f"最大推理时间: {max_inference_time:.3f} ms\n")
        f.write(f"\n===== 内存使用统计 =====\n")
        f.write(f"初始 RSS: {initial_rss:.2f} MB\n")
        f.write(f"最终 RSS: {final_rss:.2f} MB\n")
        f.write(f"平均 RSS: {avg_rss:.2f} MB\n")
        f.write(f"峰值 RSS: {peak_rss:.2f} MB\n")
        f.write(f"最小 RSS: {min_rss:.2f} MB\n")
        f.write(f"RSS Drift: {rss_drift:.2f} MB\n")
        f.write(f"RSS 波动范围: {rss_range:.2f} MB ({rss_range_percent:.2f}%)\n")
    print("结果保存成功!")
except Exception as e:
    print(f"保存结果时出错: {e}")

# 保存RSS曲线数据
rss_data_path = os.path.join(current_dir, '..', '..', 'results', 'python_rss_curve.csv')
print(f"保存RSS曲线数据到: {rss_data_path}")
try:
    with open(rss_data_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入CSV头部
        writer.writerow(['Timestamp', 'Elapsed_Seconds', 'RSS_MB'])
        
        # 写入RSS采样数据
        for sample in rss_samples:
            writer.writerow([
                sample['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                f"{sample['elapsed']:.3f}",
                f"{sample['rss']:.2f}"
            ])
    print(f"RSS曲线数据已保存: {len(rss_samples)} 个采样点")
except Exception as e:
    print(f"保存RSS曲线数据时出错: {e}")

print("\n测试完成!")
