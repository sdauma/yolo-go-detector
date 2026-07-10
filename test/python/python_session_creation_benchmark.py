# -*- coding: utf-8 -*-
# python_session_creation_benchmark.py
# Python Session鍒涘缓鏃堕棿娴嬭瘯
#
# 鎶€鏈鏄庯細
# - 浣跨敤 Python baseline Session 鎺ュ彛锛圛nferenceSession锛?
# - 閫氳繃 SessionOptions 鏄惧紡閰嶇疆threads鍙傛暟锛坕ntraOp=12, interOp=1锛?
# - 寰幆鍒涘缓100娆?InferenceSession锛屾瘡娆″垱寤哄悗 del sess 閲婃斁璧勬簮
# - 涓嶇粦瀹欳PU鏍稿績锛岃绯荤粺鑷敱璋冨害
#
# 娴嬭瘯鐩殑锛?
# - 娴嬮噺Python鍒涘缓InferenceSession鐨勬椂闂?
# - 涓嶨o鐨凷ession鍒涘缓鏃堕棿杩涜瀵规瘮
# - 鎻愪緵瀹㈣鐨勮法璇█瀵规瘮鏁版嵁

import onnxruntime as ort
import numpy as np
import time
import os
import sys
import psutil
from dataclasses import dataclass

# 鍥哄畾闅忔満绉嶅瓙锛岀‘淇濆彲澶嶇幇
np.random.seed(12345)

# 鑾峰彇褰撳墠宸ヤ綔鐩綍
current_dir = os.path.dirname(os.path.abspath(__file__))

# 鏋勫缓model璺緞
yolo11x_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
yolo11n_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# 鏋勫缓椤圭洰鏍硅矾寰?
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

@dataclass
class SessionCreationResult:
    avg_time: float
    std_time: float
    p50_time: float
    p90_time: float
    min_time: float
    max_time: float
    times: list

def run_session_creation_benchmark(model_name, model_path):
    print(f"===== Python Session鍒涘缓鏃堕棿娴嬭瘯 - {model_name} ====")
    
    # 涓嶇粦瀹欳PU鏍稿績锛岃绯荤粺鑷敱璋冨害锛堝尮閰岹o鐨勯粯璁よ涓猴級
    process = psutil.Process(os.getpid())
    print("CPU affinity: system default")
    
    # 娴嬭瘯Session鍒涘缓鏃堕棿
    print(f"娴嬭瘯{model_name}model鐨凷ession鍒涘缓鏃堕棿...")
    runs = 100  # 鍒涘缓100娆ession
    times = []

    for i in range(runs):
        t0 = time.perf_counter()
        try:
            sess_options = ort.SessionOptions()
            # threads閰嶇疆 - 12threads锛屽尮閰岹o鐨勯粯璁よ涓?
            sess_options.intra_op_num_threads = 12
            sess_options.inter_op_num_threads = 1
            # 鏃ュ織閰嶇疆锛堝叧闂墍鏈夋棩蹇楋級
            sess_options.log_severity_level = 3
            # 鎬ц兘鍒嗘瀽閰嶇疆锛堝叧闂€ц兘鍒嗘瀽锛?
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # 鍐呭瓨姹犻厤缃紙鍚敤鍐呭瓨姹犲鐢級
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            print(f"Error: Failed to create InferenceSession: {e}")
            sys.exit(1)
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000  # 杞崲涓烘绉?
        times.append(dt)
        
        # 閲婃斁Session璧勬簮
        del sess

    # Calculate results
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)
    p50_time = np.percentile(times, 50)
    p90_time = np.percentile(times, 90)

    return SessionCreationResult(
        avg_time=avg_time,
        std_time=std_time,
        p50_time=p50_time,
        p90_time=p90_time,
        min_time=min_time,
        max_time=max_time,
        times=times
    )

def main():
    print("===== Python Session Creation Time Test =====")
    print("Test configuration:")
    print("- threads: 12")
    print("- creation count: 100")
    print()

    # 娴嬭瘯 YOLO11x model
    print("\n===== Testing YOLO11x Model =====")
    if not os.path.exists(yolo11x_path):
        print(f"閿欒: YOLO11xmodel鏂囦欢涓嶅瓨鍦? {yolo11x_path}")
        sys.exit(1)
    
    yolo11x_result = run_session_creation_benchmark("YOLO11x", yolo11x_path)
    
    print(f"\nYOLO11x Session鍒涘缓鏃堕棿缁撴灉:")
    print(f"骞冲潎鏃堕棿: {yolo11x_result.avg_time:.3f} ms")
    print(f"鏍囧噯宸? {yolo11x_result.std_time:.3f} ms")
    print(f"P50鏃堕棿: {yolo11x_result.p50_time:.3f} ms")
    print(f"P90鏃堕棿: {yolo11x_result.p90_time:.3f} ms")
    print(f"鏈€灏忔椂闂? {yolo11x_result.min_time:.3f} ms")
    print(f"鏈€澶ф椂闂? {yolo11x_result.max_time:.3f} ms")

    # 娴嬭瘯 YOLO11n model
    print("\n===== Testing YOLO11n Model =====")
    if not os.path.exists(yolo11n_path):
        print(f"閿欒: YOLO11nmodel鏂囦欢涓嶅瓨鍦? {yolo11n_path}")
        sys.exit(1)
    
    yolo11n_result = run_session_creation_benchmark("YOLO11n", yolo11n_path)
    
    print(f"\nYOLO11n Session鍒涘缓鏃堕棿缁撴灉:")
    print(f"骞冲潎鏃堕棿: {yolo11n_result.avg_time:.3f} ms")
    print(f"鏍囧噯宸? {yolo11n_result.std_time:.3f} ms")
    print(f"P50鏃堕棿: {yolo11n_result.p50_time:.3f} ms")
    print(f"P90鏃堕棿: {yolo11n_result.p90_time:.3f} ms")
    print(f"鏈€灏忔椂闂? {yolo11n_result.min_time:.3f} ms")
    print(f"鏈€澶ф椂闂? {yolo11n_result.max_time:.3f} ms")

    # 淇濆瓨缁撴灉
    result_path = os.path.join(base_path, "results", "python_session_creation_result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Session鍒涘缓鏃堕棿娴嬭瘯缁撴灉 =====\n")
        f.write("娴嬭瘯閰嶇疆锛歕n")
        f.write("- threads鏁? 12\n")
        f.write("- 鍒涘缓娆℃暟: 100娆n")
        f.write("\n")
        
        f.write("===== YOLO11x 娴嬭瘯缁撴灉 =====\n")
        f.write(f"骞冲潎鏃堕棿: {yolo11x_result.avg_time:.5f} ms\n")
        f.write(f"鏍囧噯宸? {yolo11x_result.std_time:.5f} ms\n")
        f.write(f"P50鏃堕棿: {yolo11x_result.p50_time:.5f} ms\n")
        f.write(f"P90鏃堕棿: {yolo11x_result.p90_time:.5f} ms\n")
        f.write(f"鏈€灏忔椂闂? {yolo11x_result.min_time:.5f} ms\n")
        f.write(f"鏈€澶ф椂闂? {yolo11x_result.max_time:.5f} ms\n")
        f.write("\n")
        
        f.write("===== YOLO11n 娴嬭瘯缁撴灉 =====\n")
        f.write(f"骞冲潎鏃堕棿: {yolo11n_result.avg_time:.5f} ms\n")
        f.write(f"鏍囧噯宸? {yolo11n_result.std_time:.5f} ms\n")
        f.write(f"P50鏃堕棿: {yolo11n_result.p50_time:.5f} ms\n")
        f.write(f"P90鏃堕棿: {yolo11n_result.p90_time:.5f} ms\n")
        f.write(f"鏈€灏忔椂闂? {yolo11n_result.min_time:.5f} ms\n")
        f.write(f"鏈€澶ф椂闂? {yolo11n_result.max_time:.5f} ms\n")

    print(f"\nResults saved to: {result_path}")
    print("Test completed!")

if __name__ == "__main__":
    main()

