import matplotlib.pyplot as plt
import numpy as np
import os

# 导入字体配置工具
from font_utils import setup_fonts, print_font_info

# 设置字体（按照《计算机工程》期刊要求）
setup_fonts()
print_font_info()

# 获取项目根目录
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results")

# 从文件中读取内存数据
def read_memory_data(file_path):
    """从架构对比测试结果文件读取内存数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            is_shared = False
            if '架构=Shared' in line:
                is_shared = True
            elif 'architecture=Shared' in line:
                is_shared = True
            if not is_shared:
                continue
            parts = line.split(',')
            concurrency = None
            memory_str = None
            for part in parts:
                part = part.strip()
                if part.startswith('concurrency=') or part.startswith('并发=') or part.startswith('并发数='):
                    concurrency = int(part.split('=')[1].strip())
                if part.startswith('peak_rss=') or part.startswith('峰值RSS=') or part.startswith('峰值PM='):
                    memory_str = part.split('=', 1)[1].strip()
            if concurrency is None:
                concurrency = int(parts[1].split('=')[1].strip())
            if memory_str is None:
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
            elif '池大小' in line and (':' in line or '：' in line):
                # 提取并发数，支持"池大小：12"或"池大小:12"
                for sep in [':', '：']:
                    if sep in line:
                        try:
                            concurrency = int(line.split(sep)[1].strip())
                            break
                        except ValueError:
                            pass
            elif ('峰值RSS' in line or '峰值PM' in line or 'Peak PM' in line) and current_arch and concurrency is not None:
                # 提取内存值，支持"峰值RSS: 60.75000 MB"或"峰值PM: 60.75000 MB"
                for sep in [':', '：']:
                    if sep in line:
                        try:
                            memory_str = line.split(sep)[1].strip()
                            # 去掉 MB 单位
                            memory = float(memory_str.split(' ')[0])
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
    python_memory_data = read_memory_data(os.path.join(results_dir, 'python_architecture_comparison.txt'))
    go_memory_data = read_go_memory_data(os.path.join(results_dir, 'go_architecture_comparison.txt'))
except Exception as e:
    print(f"错误：{e}")
    raise

# 提取数据点
concurrency = [1, 2, 4, 8, 12]

# 从Python数据中提取内存占用
python_memory = []
for c in concurrency:
    for item in python_memory_data:
        if item[0] == c:
            python_memory.append(item[1])
            break

# 从Go数据中提取内存占用
go_memory = []
for c in concurrency:
    for item in go_memory_data:
        if item[0] == c:
            go_memory.append(item[1])
            break

# 绘图
fig, ax = plt.subplots(figsize=(7, 5))

# 绘制柱状图
width = 0.35
x = np.array(concurrency)
rects1 = ax.bar(x - width/2, python_memory, width, label='Python', color='white', edgecolor='k', hatch='/')
rects2 = ax.bar(x + width/2, go_memory, width, label='Go', color='gray', edgecolor='k', hatch='\\')

# 标签与标题
ax.set_xlabel('并发数', fontsize=11)
ax.set_ylabel('内存占用 (MB)', fontsize=11)
ax.set_title('不同并发数下的内存占用对比', fontsize=12)
ax.set_xticks(concurrency)
ax.legend(fontsize=10)
ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.6)

# 在柱上显示数值
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}', 
                    xy=(rect.get_x() + rect.get_width()/2, height), 
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# 保存图片
plt.savefig("memory_scalability.png", dpi=600, bbox_inches='tight', format='png')