import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从文件中读取内存数据
def read_memory_data(file_path):
    """从架构对比测试结果文件读取内存数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_arch = None
        concurrency = None
        for line in lines:
            line = line.strip()
            if '架构=Shared' in line:
                parts = line.split(',')
                concurrency = int(parts[1].split('=')[1].strip())
                memory_str = parts[-2].split('=')[1].strip()
                memory = float(memory_str.split(' ')[0])
                data.append((concurrency, memory))
        
        if not data:
            raise ValueError(f"未从文件 {file_path} 中读取到任何内存数据")
            
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取内存数据文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取内存数据失败：{e}")

# 从Go文件中读取内存数据
def read_go_memory_data(file_path):
    """从 Go 架构对比测试结果文件读取内存数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_arch = None
        concurrency = None
        for line in lines:
            line = line.strip()
            if '===== Session Pool =====' in line:
                current_arch = 'session_pool'
            elif '池大小:' in line or '池大小:' in line:
                for sep in [':', ':']:
                    if sep in line:
                        try:
                            concurrency = int(line.split(sep)[1].strip())
                            break
                        except ValueError:
                            pass
            elif ('峰值 RSS:' in line or 'Peak RSS:' in line) and current_arch:
                for sep in [':', ':']:
                    if sep in line:
                        try:
                            memory = float(line.split(sep)[1].split(' ')[0].strip())
                            data.append((concurrency, memory))
                            break
                        except ValueError:
                            pass
        
        if not data:
            raise ValueError(f"未从文件 {file_path} 中读取到任何 Go 内存数据")
            
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取 Go 内存数据文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取 Go 内存数据失败：{e}")

# 读取数据
try:
    python_memory_data = read_memory_data('../../results/python_architecture_comparison.txt')
    go_memory_data = read_go_memory_data('../../results/go_architecture_comparison.txt')
    
    concurrency = [1, 4, 8, 12]
    
    python_memory = []
    for c in concurrency:
        found = False
        for item in python_memory_data:
            if item[0] == c:
                python_memory.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Python 数据缺少并发数：{c}")
    
    go_memory = []
    for c in concurrency:
        found = False
        for item in go_memory_data:
            if item[0] == c:
                go_memory.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Go 数据缺少并发数：{c}")
except Exception as e:
    print(f"错误：{e}")
    raise

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制内存占用曲线
ax.plot(concurrency, python_memory, 'k--o', label='Python 传统部署')
ax.plot(concurrency, go_memory, 'k-.s', label='Go Session 池', linewidth=2)

# 设置标签和标题
ax.set_xlabel('并发数', fontsize=11)
ax.set_ylabel('内存占用 (MB)', fontsize=11)
ax.set_title('不同并发数下的内存占用对比', fontsize=12)

# 设置x轴刻度
ax.set_xticks(concurrency)

# 添加网格
ax.grid(True, linestyle=':', color='gray', alpha=0.6)

# 添加图例
ax.legend(fontsize=10)

# 保存图片
plt.savefig("memory_usage_comparison.png", dpi=600, bbox_inches='tight', format='png')
plt.show()