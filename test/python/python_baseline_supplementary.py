# -*- coding: utf-8 -*-
# python_baseline_supplementary.py
# Python 鍩哄噯娴嬭瘯琛ュ厖 - 澶歵hreads閰嶇疆瀵规瘮
#
# 鎶€鏈鏄庯細
# - 浣跨敤 Python baseline Session 鎺ュ彛锛圛nferenceSession锛?
# - 閫氳繃 SessionOptions 鏄惧紡閰嶇疆 intra_op_num_threads锛?/2/4/8锛?
# - inter_op_num_threads 鍥哄畾涓?1
# - 涓嶅惎鐢?I/O Binding锛屼娇鐢?sess.run() 鏍囧噯璋冪敤璺緞
# - Python ONNX Runtime 鐨?run() 鏂规硶姣忔璋冪敤鏃惰繘琛屾暟鎹嫹璐?
#
# 娴嬭瘯鐩殑锛?
# - 琛ュ厖鍩哄噯娴嬭瘯鐨則hreads鎵╁睍鎬ф暟鎹?
# - 娴嬭瘯涓嶅悓threads閰嶇疆涓嬬殑鎬ц兘宸紓
# - 涓?Go 绔熀鍑嗘祴璇曞榻?

import onnxruntime as ort
import numpy as np
import time
import os
import psutil

def get_process_rss():
    """杩斿洖杩涚▼绉佹湁鍐呭瓨锛圥rivateMemorySize64锛夛紝涓?Go 绔?memutil.PrivateMemoryMB() 瀵归綈"""
    process = psutil.Process(os.getpid())
    return process.memory_info().private / 1024 / 1024  # MB

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
    print(f"===== 瀹為獙缂栧彿 S-B{num_threads}: intra_op_num_threads={num_threads} =====")
    print("鎵ц璺緞锛欱aseline InferenceSession锛堜笉鍚敤 io_binding锛屼笉棰勫垎閰嶈緭鍑猴級")
    
    # 璁＄畻椤圭洰鏍硅矾寰?
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(os.path.dirname(current_dir))
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print("Creating InferenceSession...")
    try:
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        print("InferenceSession created successfully!")
    except Exception as e:
        print(f"Error: Failed to create InferenceSession: {e}")
        return None, None
    
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    
    print(f"杈撳叆鍚嶇О: {input_name}")
    print(f"杈撳叆褰㈢姸: {input_shape}")
    
    print("浠庨鐢熸垚鐨勪簩杩涘埗鏂囦欢Loading input data...")
    input_data_path = os.path.join(base_path, "test", "data", "input_data.bin")
    try:
        input_data = np.fromfile(input_data_path, dtype=np.float32).reshape(input_shape)
        print(f"Input data loaded successfully: {input_data_path}")
    except Exception as e:
        print(f"Failed to load input data: {e}")
        return None, None
    
    start_rss = get_process_rss()
    print(f"Start PM: {start_rss:.5f} MB")
    
    print("Warming up...")
    for i in range(10):
        outputs = sess.run(None, {input_name: input_data})
    
    warmup_rss = get_process_rss()
    print(f"Warmup 鍚?RSS: {warmup_rss:.5f} MB")
    
    print("寮€濮嬪熀鍑嗘祴璇?..")
    latencies = []
    for i in range(100):
        start = time.perf_counter()
        outputs = sess.run(None, {input_name: input_data})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    peak_rss = get_process_rss()
    print(f"Peak PM: {peak_rss:.5f} MB")
    
    metrics = calculate_metrics(latencies)
    
    print(f"鎬ц兘鎸囨爣: avg={metrics['avg']:.5f} ms, p50={metrics['p50']:.5f} ms, "
          f"p90={metrics['p90']:.5f} ms, p99={metrics['p99']:.5f} ms, "
          f"min={metrics['min']:.5f} ms, max={metrics['max']:.5f} ms")
    
    engineering_metrics = {
        'tensor_allocation_count': 'N/A (baseline)',
        'io_binding_enabled': False,
        'session_creation_count': 1,
        'peak_rss': peak_rss
    }
    
    print(f"宸ョ▼鎸囨爣: Tensor鍒嗛厤娆℃暟={engineering_metrics['tensor_allocation_count']}, "
          f"I/O Binding={engineering_metrics['io_binding_enabled']}, "
          f"Session鍒涘缓娆℃暟={engineering_metrics['session_creation_count']}, "
          f"peak_rss={engineering_metrics['peak_rss']:.5f} MB")
    
    return metrics, engineering_metrics

def main():
    print("===== Python Baseline Supplementary Experiment =====")
    print("Experiment purpose: Engineering-level performance evaluation comparison (non-language layer capability comparison)")
    print("Comparison strategy: Python still uses baseline (no io_binding, no pre-allocated output)")
    print("Reason: Python's io_binding execution highly depends on implementation and fixed mode")
    print("Difficult to guarantee complete consistency with Go at the engineering level, so not included in supplementary experiments")
    print()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, "third_party", "yolo11x.onnx")
    
    print(f"褰撳墠鐩綍: {current_dir}")
    print(f"椤圭洰鏍硅矾寰? {project_root}")
    print(f"model璺緞: {model_path}")
    print()
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
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
    print("===== Supplementary Experiment Completed =====")

def save_results(results, engineering_results):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    result_path = os.path.join(project_root, "results", "python_baseline_supplementary.txt")
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Python Baseline 琛ュ厖瀹為獙缁撴灉 =====\n")
        f.write("瀹為獙鎬ц川锛氬伐绋嬬骇鎺ュ彛鑳藉姏璇勪及瀵圭収锛堥潪璇█绾ф€ц兘姣旇緝锛塡n")
        f.write("鎵ц璺緞锛欱aseline InferenceSession锛堜笉鍚敤 io_binding锛屼笉棰勫垎閰嶈緭鍑猴級\n")
        f.write("瀵圭収绛栫暐锛歅ython 浠嶄娇鐢?baseline锛堜笉鍚敤 io_binding锛塡n")
        f.write("鍘熷洜锛歅ython 渚?io_binding 鐨勮涓洪珮搴︿緷璧栫増鏈笌缁戝畾鏂瑰紡锛孿n")
        f.write("闅句互鍦ㄥ伐绋嬪眰闈繚璇佷笌 Go 瀹屽叏涓€鑷达紝鍥犳鏈撼鍏ヨˉ鍏呭疄楠屽鐓с€俓n\n")
        
        f.write("鎬ц兘鎸囨爣锛歕n")
        f.write("threads閰嶇疆\tavg_latency\tP50\tP90\tP99\t鏈€灏忓€糪t鏈€澶у€糪n")
        for num_threads in [1, 2, 4, 8]:
            if num_threads in results:
                metrics = results[num_threads]
                f.write(f"{num_threads}\t{metrics['avg']:.5f}\t{metrics['p50']:.5f}\t"
                       f"{metrics['p90']:.5f}\t{metrics['p99']:.5f}\t"
                       f"{metrics['min']:.5f}\t{metrics['max']:.5f}\n")
        
        f.write("\n宸ョ▼鎸囨爣锛歕n")
        f.write("threads閰嶇疆\tTensor鍒嗛厤娆℃暟\tI/O Binding\tSession鍒涘缓娆℃暟\tpeak_rss(MB)\n")
        for num_threads in [1, 2, 4, 8]:
            if num_threads in engineering_results:
                metrics = engineering_results[num_threads]
                f.write(f"{num_threads}\t{metrics['tensor_allocation_count']}\t"
                       f"{metrics['io_binding_enabled']}\t{metrics['session_creation_count']}\t"
                       f"{metrics['peak_rss']:.5f}\n")
        
        f.write("\n涓嶅彲姣斿０鏄庯細\n")
        f.write("鏈妭瀹為獙閫氳繃 AdvancedSession 涓?I/O Binding 寮曞叆浜嗗伐绋嬬骇鎵ц璺緞浼樺寲锛孿n")
        f.write("鍏跺唴瀛樺垎閰嶅拰鎵ц璋冨害鏈哄埗涓庡墠鏂?baseline 娴嬭瘯瀛樺湪鏈川宸紓锛孿n")
        f.write("鍥犳缁撴灉涓嶇敤浜庝慨姝ｈ瑷€绾ф€ц兘缁撹锛屼粎鐢ㄤ簬璇勪及 Go 鍦?ONNX 鎺ㄧ悊浠诲姟涓殑宸ョ▼鎺ュ彛鎬ц兘娼滃姏銆俓n")
    
    print(f"Results saved to: {result_path}")

if __name__ == "__main__":
    main()

