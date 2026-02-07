#!/usr/bin/env python3
# generate_input_data.py
# 生成统一的输入数据文件，确保 Go 和 Python 版本使用完全相同的输入数据

import numpy as np
import os

# 设置固定随机种子
np.random.seed(12345)

# 生成输入数据
input_shape = (1, 3, 640, 640)
input_data = np.random.rand(*input_shape).astype(np.float32)

# 计算数据大小
data_size = input_data.size * input_data.itemsize
print(f"Input data generated")
print(f"Shape: {input_shape}")
print(f"Data type: {input_data.dtype}")
print(f"Size: {data_size / 1024 / 1024:.2f} MB")
print(f"Min value: {input_data.min():.6f}")
print(f"Max value: {input_data.max():.6f}")
print(f"Mean value: {input_data.mean():.6f}")

# 保存为二进制文件
output_dir = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "input_data.bin")
input_data.tofile(output_path)
print(f"\nInput data saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# 验证文件是否正确保存
try:
    loaded_data = np.fromfile(output_path, dtype=np.float32).reshape(input_shape)
    print(f"\nFile validation successful!")
    print(f"Loaded shape: {loaded_data.shape}")
    print(f"Loaded min value: {loaded_data.min():.6f}")
    print(f"Loaded max value: {loaded_data.max():.6f}")
    print(f"Loaded mean value: {loaded_data.mean():.6f}")
    
    # 验证数据是否一致
    if np.array_equal(input_data, loaded_data):
        print("✓ Data integrity verified: loaded data matches original")
    else:
        print("✗ Data integrity check failed: loaded data differs from original")
        
except Exception as e:
    print(f"✗ File validation failed: {e}")

print("\nInput data generation complete!")
print("\nNext steps:")
print("1. Run this script to generate input_data.bin")
print("2. Update Go and Python test files to load this input data")
print("3. Ensure both versions use the same input data for fair comparison")