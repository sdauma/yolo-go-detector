#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体配置工具模块
按照《计算机系统应用》期刊要求设置字体：
- 中文：华文中宋
- 英文：Times New Roman
"""

import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*iCCP.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')
from matplotlib import font_manager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_fonts():
    """设置字体配置，按照《计算机系统应用》期刊要求使用华文中宋"""
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 字体文件可能存在的路径
    font_paths = [
        os.path.join(base_dir, "paper", "STZhongsong.ttf"),  # 英文文件名副本：华文中宋(ASCII内部名)，避免PDF中文资源名报错
        os.path.join(base_dir, "paper", "华文中宋.ttf"),      # 原始中文文件名（兜底）
        "C:\\Windows\\Fonts\\STZHONGS.TTF",                  # 系统华文中宋（回退）
        "C:\\Windows\\Fonts\\simsun.ttc",                    # 系统宋体（回退）
        "C:\\Users\\Administrator\\AppData\\Local\\Microsoft\\Windows\\Fonts\\simsun.ttc"  # 用户字体目录（回退）
    ]
    
    # 尝试注册字体
    font_registered = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            # 添加到字体管理器
            font_manager.fontManager.addfont(font_path)
            font_registered = True
            break
    
    # 设置字体：中文使用华文中宋（STZhongsong），英文使用 Times New Roman
    plt.rcParams['font.sans-serif'] = ['STZhongsong', 'SimSun', 'SimHei']  # 与 fig1 统一为华文中宋，回退宋体/黑体
    plt.rcParams['font.serif'] = ['Times New Roman']  # 英文使用 Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    return font_registered

def get_available_fonts():
    """获取可用的字体信息"""
    font_names = [f.name for f in font_manager.fontManager.ttflist]
    chinese_fonts = [name for name in font_names if 'STZhongsong' in name or 'SimHei' in name or 'SimSun' in name]
    english_fonts = [name for name in font_names if 'Times' in name]
    return chinese_fonts, english_fonts

def print_font_info():
    """打印字体配置信息"""
    from matplotlib.font_manager import findfont, FontProperties
    
    # 获取实际使用的字体文件路径
    chinese_font_path = findfont(FontProperties(family=plt.rcParams['font.sans-serif']))
    english_font_path = findfont(FontProperties(family=plt.rcParams['font.serif']))
    
    print("\n=== 字体配置信息 ===")
    print("中文字体 (sans-serif):", plt.rcParams['font.sans-serif'])
    print("英文字体 (serif):", plt.rcParams['font.serif'])
    
    chinese_fonts, english_fonts = get_available_fonts()
    print("可用的中文字体:", chinese_fonts[:5] if chinese_fonts else "未找到特定中文字体")
    print("可用的Times字体:", english_fonts[:5] if english_fonts else "未找到Times New Roman")
    
    print("\n=== 实际使用的字体 ===")
    print("中文字体文件路径:", chinese_font_path)
    print("英文字体文件路径:", english_font_path)
    
    # 从路径中提取字体名称
    chinese_font_name = os.path.basename(chinese_font_path).lower()
    english_font_name = os.path.basename(english_font_path).lower()
    
    if 'stzhongsong' in chinese_font_name or 'huawen' in chinese_font_name or '华文中宋' in chinese_font_path:
        print("[OK] 中文实际使用: 华文中宋 (STZhongsong)")
    elif 'simhei' in chinese_font_name:
        print("[OK] 中文实际使用: 黑体 (SimHei) - 回退字体")
    elif 'simsun' in chinese_font_name:
        print("[OK] 中文实际使用: 宋体 (SimSun) - 回退字体")
    else:
        print("[?] 中文实际使用:", os.path.basename(chinese_font_path))
    
    if 'times' in english_font_name:
        print("[OK] 英文实际使用: Times New Roman")
    else:
        print("[?] 英文实际使用:", os.path.basename(english_font_path))
    print("=====================\n")
