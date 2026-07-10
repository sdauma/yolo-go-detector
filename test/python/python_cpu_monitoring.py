# -*- coding: utf-8 -*-
# python_cpu_monitoring.py
# Python CPU 鐩戞帶娴嬭瘯 - 鐩戞祴鎺ㄧ悊杩囩▼涓殑 CPU 浣跨敤鐜?
#
# 鎶€鏈鏄庯細
# - 浣跨敤 Python baseline Session 鎺ュ彛锛圛nferenceSession锛?
# - 閫氳繃 SessionOptions 鏄惧紡閰嶇疆threads鍙傛暟锛坕ntraOp=1, interOp=1锛?
# - 浣跨敤 sess.run() 鏍囧噯璋冪敤璺緞锛屼笉鍚敤 I/O Binding
# - 浣跨敤 psutil 閲囨牱鎺ㄧ悊鍓嶅悗鐨?CPU 浣跨敤鐜?
#
# 娴嬭瘯鐩殑锛?
# - 鐩戞祴 ONNX Runtime 鎺ㄧ悊鏃剁殑 CPU 浣跨敤鐜?
# - 鍒嗘瀽涓嶅悓concurrency閰嶇疆涓嬬殑 CPU 璐熻浇鍒嗗竷
# - 涓鸿鏂囨彁渚?CPU 鍒╃敤鐜囨暟鎹?

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass
from typing import List
import threading

# 鍥哄畾闅忔満绉嶅瓙锛岀‘淇濆彲澶嶇幇
np.random.seed(12345)

# 鑾峰彇褰撳墠宸ヤ綔鐩綍
current_dir = os.path.dirname(os.path.abspath(__file__))

# 鏋勫缓model璺緞
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))

# 鏋勫缓椤圭洰鏍硅矾寰?
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 妫€鏌odel鏂囦欢鏄惁瀛樺湪
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

print("===== Python CPU Monitoring Test =====")
print(f"model璺緞: {model_path}")

# 鍒涘缓杈撳叆鏁版嵁
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 鑾峰彇 CPU 淇℃伅
print(f"\n绯荤粺 CPU 淇℃伅:")
print(f"  鐗╃悊鏍稿績鏁? {psutil.cpu_count(logical=False)}")
print(f"  閫昏緫鏍稿績鏁? {psutil.cpu_count(logical=True)}")
print(f"  褰撳墠 CPU 棰戠巼: {psutil.cpu_freq().current:.0f} MHz")

# Create Session
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

print(f"\nCreating InferenceSession...")
session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 棰勭儹
print("Warming up...")
for _ in range(5):
    session.run(None, {input_name: input_data})

# 鐩戞祴 CPU 浣跨敤鐜?
print("\nStarting CPU usage monitoring...")
num_requests = 50
cpu_samples = []

# 鑾峰彇鍒濆 CPU 浣跨敤鐜?
process = psutil.Process()
initial_cpu_percent = process.cpu_percent()
initial_time = time.time()

for i in range(num_requests):
    # 璁板綍鎺ㄧ悊鍓嶇殑 CPU 浣跨敤鐜?
    cpu_before = process.cpu_percent()
    
    # 鎵ц鎺ㄧ悊
    start = time.perf_counter()
    session.run(None, {input_name: input_data})
    latency = (time.perf_counter() - start) * 1000  # ms
    
    # 璁板綍鎺ㄧ悊鍚庣殑 CPU 浣跨敤鐜?
    cpu_after = process.cpu_percent()
    
    cpu_samples.append({
        'request': i + 1,
        'latency': latency,
        'cpu_before': cpu_before,
        'cpu_after': cpu_after
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1}/{num_requests} requests")

# 璁＄畻缁熻淇℃伅
latencies = [s['latency'] for s in cpu_samples]
cpu_afters = [s['cpu_after'] for s in cpu_samples]

avg_latency = sum(latencies) / len(latencies)
avg_cpu = sum(cpu_afters) / len(cpu_afters)
max_cpu = max(cpu_afters)
min_cpu = min(cpu_afters)

print(f"\n===== Test Results =====")
print(f"total_requests: {num_requests}")
print(f"avg_latency: {avg_latency:.2f} ms")
print(f"avg_cpu: {avg_cpu:.2f}%")
print(f"max_cpu: {max_cpu:.2f}%")
print(f"min_cpu: {min_cpu:.2f}%")

# 淇濆瓨缁撴灉
result_path = os.path.join(base_path, "results", "python_cpu_monitoring_result.txt")
os.makedirs(os.path.dirname(result_path), exist_ok=True)

with open(result_path, 'w', encoding='utf-8') as f:
    f.write("===== Python CPU 鐩戞帶娴嬭瘯缁撴灉 =====\n\n")
    f.write(f"model: YOLO11x\n")
    f.write(f"杈撳叆灏哄: 1x3x640x640\n")
    f.write(f"intra_op_num_threads: 1\n")
    f.write(f"inter_op_num_threads: 1\n\n")
    f.write(f"绯荤粺淇℃伅:\n")
    f.write(f"  鐗╃悊鏍稿績鏁? {psutil.cpu_count(logical=False)}\n")
    f.write(f"  閫昏緫鏍稿績鏁? {psutil.cpu_count(logical=True)}\n")
    f.write(f"  褰撳墠 CPU 棰戠巼: {psutil.cpu_freq().current:.0f} MHz\n\n")
    f.write(f"鎬ц兘鎸囨爣:\n")
    f.write(f"  total_requests: {num_requests}\n")
    f.write(f"  avg_latency: {avg_latency:.2f} ms\n")
    f.write(f"  avg_cpu: {avg_cpu:.2f}%\n")
    f.write(f"  max_cpu: {max_cpu:.2f}%\n")
    f.write(f"  min_cpu: {min_cpu:.2f}%\n\n")
    f.write("璇︾粏鏁版嵁:\n")
    f.write("璇锋眰鍙? 寤惰繜(ms), cpu_usage(%)\n")
    for s in cpu_samples:
        f.write(f"{s['request']}, {s['latency']:.2f}, {s['cpu_after']:.2f}\n")

print(f"\nResults saved to: {result_path}")
print("===== Test Completed =====")

