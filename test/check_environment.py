#!/usr/bin/env python3
# check_environment.py
# 检查环境信息，确保测试环境的一致性和可复现性

import os
import platform
import sys
import subprocess

# 主函数
def main():
    # 获取项目根目录
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    
    # 结果文件路径
    output_dir = os.path.join(project_root, "results")
    output_path = os.path.join(output_dir, "env_check_result.txt")
    
    # 确保结果目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集环境信息
    env_info = {
        "系统信息": {
            "操作系统": platform.platform(),
            "系统版本": platform.version(),
            "架构": platform.architecture(),
            "处理器": platform.processor(),
            "Python版本": platform.python_version(),
            "Python实现": platform.python_implementation(),
        },
        "硬件信息": {
            "CPU信息": get_cpu_info(),
            "内存信息": get_memory_info(),
        },
        "Go信息": {
            "Go版本": get_go_version(),
        },
        "ONNX Runtime信息": {
            "Python ONNX Runtime版本": get_onnxruntime_version(),
        },
        "项目信息": {
            "项目路径": project_root,
            "工作目录": os.getcwd(),
            "环境变量": get_relevant_env_vars(),
        },
    }
    
    # 保存环境信息
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 环境检查结果\n")
        f.write(f"# 生成时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
        f.write("\n")
        
        for section, info in env_info.items():
            f.write(f"## {section}\n")
            for key, value in info.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
    
    print(f"✓ 环境检查结果已保存到: {output_path}")

# 获取CPU信息
def get_cpu_info():
    """获取CPU信息"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, check=True)
            return result.stdout.strip().split("\n")[1].strip()
        else:
            return platform.processor()
    except Exception as e:
        return f"获取失败: {e}"

# 获取内存信息
def get_memory_info():
    """获取内存信息"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"], capture_output=True, text=True, check=True)
            memory_bytes = int(result.stdout.strip().split("\n")[1].strip())
            memory_gb = memory_bytes / (1024 ** 3)
            return f"{memory_gb:.2f} GB"
        else:
            return "获取失败: 不支持的操作系统"
    except Exception as e:
        return f"获取失败: {e}"

# 获取Go版本
def get_go_version():
    """获取Go版本"""
    try:
        result = subprocess.run(["go", "version"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        return f"获取失败: {e}"

# 获取ONNX Runtime版本
def get_onnxruntime_version():
    """获取ONNX Runtime版本"""
    try:
        import onnxruntime
        return onnxruntime.__version__
    except Exception as e:
        return f"获取失败: {e}"

# 获取相关环境变量
def get_relevant_env_vars():
    """获取相关环境变量"""
    relevant_vars = ["PATH", "GOPATH", "GOROOT"]
    env_vars = {}
    for var in relevant_vars:
        if var in os.environ:
            # 只显示前1000个字符，避免输出过长
            value = os.environ[var]
            if len(value) > 1000:
                value = value[:1000] + "..."
            env_vars[var] = value
    return env_vars

if __name__ == "__main__":
    main()
