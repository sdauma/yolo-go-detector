#!/usr/bin/env python3
# env_check.py
# 重新检查软硬件环境并生成准确的环境检查结果文件

import os
import sys
import platform
import subprocess

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..'))

# 构建结果文件路径
result_path = os.path.join(base_path, 'results', 'env_check_result.txt')

# 收集系统信息
def get_system_info():
    info = {}
    
    # 操作系统信息
    info['os'] = platform.platform()
    info['os_version'] = platform.version()
    info['os_architecture'] = platform.architecture()[0]
    
    # CPU信息
    try:
        if sys.platform == 'win32':
            output = subprocess.check_output(['wmic', 'cpu', 'get', 'name', '/Value'], 
                                           universal_newlines=True)
            for line in output.split('\n'):
                if 'Name' in line:
                    info['cpu_brand'] = line.split('=')[1].strip()
                    break
        else:
            # Linux/Mac
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu_brand'] = line.split(':')[1].strip()
                        break
    except:
        info['cpu_brand'] = 'Unknown'
    
    info['cpu_cores'] = os.cpu_count()
    
    # 内存信息
    try:
        if sys.platform == 'win32':
            output = subprocess.check_output(['wmic', 'OS', 'get', 'TotalVisibleMemorySize', '/Value'], 
                                           universal_newlines=True)
            for line in output.split('\n'):
                if 'TotalVisibleMemorySize' in line:
                    memory_kb = int(line.split('=')[1].strip())
                    info['total_memory_gb'] = round(memory_kb / 1024 / 1024, 2)
                    break
        else:
            # Linux/Mac
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_kb = int(line.split()[1])
                        info['total_memory_gb'] = round(memory_kb / 1024 / 1024, 2)
                        break
    except:
        info['total_memory_gb'] = 'Unknown'
    
    return info

# 收集Python信息
def get_python_info():
    info = {}
    info['python_version'] = platform.python_version()
    info['python_executable'] = sys.executable
    
    # 检查onnxruntime版本
    try:
        import onnxruntime
        info['onnxruntime_version'] = onnxruntime.__version__
    except ImportError:
        info['onnxruntime_version'] = 'Not installed'
    
    # 检查numpy版本
    try:
        import numpy
        info['numpy_version'] = numpy.__version__
    except ImportError:
        info['numpy_version'] = 'Not installed'
    
    # 检查opencv版本
    try:
        import cv2
        info['opencv_version'] = cv2.__version__
    except ImportError:
        info['opencv_version'] = 'Not installed'
    
    return info

# 收集Go信息
def get_go_info():
    info = {}
    try:
        output = subprocess.check_output(['go', 'version'], universal_newlines=True)
        info['go_version'] = output.strip()
    except:
        info['go_version'] = 'Not installed'
    
    return info

# 收集模型信息
def get_model_info():
    info = {}
    model_path = os.path.join(base_path, 'third_party', 'yolo11x.onnx')
    
    if os.path.exists(model_path):
        info['model_exists'] = True
        info['model_size_mb'] = round(os.path.getsize(model_path) / 1024 / 1024, 2)
    else:
        info['model_exists'] = False
        info['model_size_mb'] = 0
    
    return info

# 生成环境检查结果
def generate_env_check():
    print("===== 开始环境检查 =====")
    
    # 收集所有信息
    system_info = get_system_info()
    python_info = get_python_info()
    go_info = get_go_info()
    model_info = get_model_info()
    
    # 生成结果内容
    content = []
    content.append("===== 系统环境检查结果 =====")
    content.append(f"操作系统: {system_info['os']}")
    content.append(f"操作系统版本: {system_info['os_version']}")
    content.append(f"操作系统架构: {system_info['os_architecture']}")
    content.append(f"CPU: {system_info['cpu_brand']}")
    content.append(f"CPU核心数: {system_info['cpu_cores']}")
    content.append(f"总内存: {system_info['total_memory_gb']} GB")
    content.append("")
    
    content.append("===== Python环境检查结果 =====")
    content.append(f"Python版本: {python_info['python_version']}")
    content.append(f"Python可执行文件: {python_info['python_executable']}")
    content.append(f"ONNX Runtime版本: {python_info['onnxruntime_version']}")
    content.append(f"NumPy版本: {python_info['numpy_version']}")
    content.append(f"OpenCV版本: {python_info['opencv_version']}")
    content.append("")
    
    content.append("===== Go环境检查结果 =====")
    content.append(f"Go版本: {go_info['go_version']}")
    content.append("")
    
    content.append("===== 模型检查结果 =====")
    content.append(f"模型文件存在: {model_info['model_exists']}")
    content.append(f"模型大小: {model_info['model_size_mb']} MB")
    content.append("")
    
    content.append("===== 环境检查完成 =====")
    
    # 写入结果文件
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    print(f"环境检查结果已保存到: {result_path}")
    print("===== 环境检查完成 =====")

if __name__ == "__main__":
    generate_env_check()