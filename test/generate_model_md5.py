#!/usr/bin/env python3
# generate_model_md5.py
# 生成模型文件的MD5校验值，确保模型文件的完整性

import os
import hashlib

# 计算文件的MD5校验值
def calculate_md5(file_path):
    """计算文件的MD5校验值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取文件，避免一次性读取大文件
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()

# 主函数
def main():
    # 获取项目根目录
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    
    # 模型文件目录
    model_dir = os.path.join(project_root, "third_party")
    
    # 结果文件路径
    output_dir = os.path.join(project_root, "results")
    output_path = os.path.join(output_dir, "model_md5.txt")
    
    # 确保结果目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 模型文件列表
    model_files = [
        "yolo11x.onnx",
        "yolov8x.onnx"
    ]
    
    # 计算MD5校验值并保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 模型文件MD5校验值\n")
        f.write(f"# 生成时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
        f.write("\n")
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                md5_value = calculate_md5(model_path)
                file_size = os.path.getsize(model_path) / 1024 / 1024  # 转换为MB
                f.write(f"## {model_file}\n")
                f.write(f"- 文件路径: {model_path}\n")
                f.write(f"- MD5校验值: {md5_value}\n")
                f.write(f"- 文件大小: {file_size:.2f} MB\n")
                f.write("\n")
                print(f"✓ 已计算 {model_file} 的MD5校验值: {md5_value}")
            else:
                f.write(f"## {model_file}\n")
                f.write(f"- 文件路径: {model_path}\n")
                f.write(f"- 状态: 文件不存在\n")
                f.write("\n")
                print(f"✗ {model_file} 文件不存在")
    
    print(f"\n✓ MD5校验结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
