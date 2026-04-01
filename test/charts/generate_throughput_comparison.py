import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 从文件中读取数据
def read_architecture_data(file_path):
    """从架构对比测试结果文件读取吞吐量数据"""
    data = {
        'unsafe_shared': [],
        'mutex_shared': [],
        'session_pool': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_arch = None
        concurrency = None
        for line in lines:
            line = line.strip()
            if line.startswith('===== Unsafe Shared ====='):
                current_arch = 'unsafe_shared'
            elif line.startswith('===== Mutex Shared ====='):
                current_arch = 'mutex_shared'
            elif line.startswith('===== Session Pool ====='):
                current_arch = 'session_pool'
            elif '并发度:' in line and current_arch:
                # 处理缩进和格式
                parts = line.split('并发度:')
                if len(parts) > 1:
                    try:
                        concurrency = int(parts[1].strip())
                    except ValueError:
                        pass
            elif '池大小:' in line and current_arch == 'session_pool':
                # 处理缩进和格式
                parts = line.split('池大小:')
                if len(parts) > 1:
                    try:
                        concurrency = int(parts[1].strip())
                    except ValueError:
                        pass
            elif '吞吐量:' in line and current_arch and concurrency is not None:
                # 处理缩进和格式
                parts = line.split('吞吐量:')
                if len(parts) > 1:
                    try:
                        throughput_str = parts[1].strip()
                        throughput = float(throughput_str.split(' ')[0])
                        data[current_arch].append((concurrency, throughput))
                        concurrency = None
                    except ValueError:
                        pass
        
        # 打印读取的数据，用于调试
        print("读取的数据:")
        for arch_key, arch_data in data.items():
            print(f"{arch_key}: {arch_data}")
        
        # 验证数据是否完整
        for arch_key in data:
            if not data[arch_key]:
                raise ValueError(f"未从文件中读取到 {arch_key} 的吞吐量数据")
        
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取架构对比数据文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取架构数据失败：{e}")

# 读取Python架构数据
def read_python_data(file_path):
    """从 Python 架构对比测试结果文件读取吞吐量数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '架构=Shared' in line:
                    parts = line.split(',')
                    concurrency = int(parts[1].split('=')[1].strip())
                    # 处理吞吐量格式，提取数字部分
                    throughput_str = parts[2].split('=')[1].strip()
                    throughput = float(throughput_str.split(' ')[0])
                    data.append((concurrency, throughput))
        
        if not data:
            raise ValueError("未从文件中读取到任何 Python 吞吐量数据")
            
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"无法读取 Python 架构数据文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取 Python 架构数据失败：{e}")

# 读取数据
try:
    # 使用绝对路径
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    go_data_file = os.path.join(project_root, 'results', 'go_architecture_comparison.txt')
    python_data_file = os.path.join(project_root, 'results', 'python_architecture_comparison.txt')
    
    go_data = read_architecture_data(go_data_file)
    python_data = read_python_data(python_data_file)
    
    concurrency = [1, 4, 8, 12]
    
    go_unsafe_shared = []
    go_mutex_shared = []
    go_session_pool = []
    
    for c in concurrency:
        found = False
        for item in go_data['unsafe_shared']:
            if item[0] == c:
                go_unsafe_shared.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Go Unsafe Shared 数据缺少并发数：{c}")
        
        found = False
        for item in go_data['mutex_shared']:
            if item[0] == c:
                go_mutex_shared.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Go Mutex Shared 数据缺少并发数：{c}")
        
        found = False
        for item in go_data['session_pool']:
            if item[0] == c:
                go_session_pool.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Go Session Pool 数据缺少并发数：{c}")
    
    python_shared = []
    for c in concurrency:
        found = False
        for item in python_data:
            if item[0] == c:
                python_shared.append(item[1])
                found = True
                break
        if not found:
            raise ValueError(f"Python Shared 数据缺少并发数：{c}")
except Exception as e:
    print(f"错误：{e}")
    raise

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制不同架构的吞吐量曲线
ax.plot(concurrency, go_unsafe_shared, 'k--o', label='Go 无锁共享')
ax.plot(concurrency, go_mutex_shared, 'k-.s', label='Go 互斥锁共享')
ax.plot(concurrency, python_shared, 'k:^', label='Python 共享')
ax.plot(concurrency, go_session_pool, 'k-o', label='Go Session 池', linewidth=2)

# 设置标签和标题
ax.set_xlabel('并发数', fontsize=11)
ax.set_ylabel('吞吐量 (REQ/s)', fontsize=11)
ax.set_title('不同并发架构的吞吐量对比', fontsize=12)

# 设置x轴刻度
ax.set_xticks(concurrency)

# 添加网格
ax.grid(True, linestyle=':', color='gray', alpha=0.6)

# 添加图例
ax.legend(fontsize=10)

# 调整y轴范围，为箭头留出空间
y_min = min(min(go_unsafe_shared), min(go_mutex_shared), min(python_shared), min(go_session_pool))
y_max = max(max(go_unsafe_shared), max(go_mutex_shared), max(python_shared), max(go_session_pool))
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)

# 保存图片
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'charts')
output_path = os.path.join(output_dir, 'fig2_throughput_comparison.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
print(f"图表已保存到: {output_path}")

# 保存到final_charts目录
final_output_dir = os.path.join(output_dir, 'final_charts')
os.makedirs(final_output_dir, exist_ok=True)
final_output_path = os.path.join(final_output_dir, 'fig2_throughput_comparison.png')
plt.savefig(final_output_path, dpi=600, bbox_inches='tight', format='png')
print(f"图表已保存到: {final_output_path}")

plt.show()