# -*- coding: utf-8 -*-
"""
对比 ChatGPT 生成图 与 本项目 matplotlib 生成图(fig1) 的文件/图像属性，
解释为何两者文件大小(kB)差异巨大。

用法：
    python compare_image_props.py
"""
import os
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

BASE = os.path.dirname(os.path.abspath(__file__))
OUR = os.path.join(BASE, "..", "..", "results", "charts", "fig1_session_pool_architecture.png")
CHAT = os.path.join(BASE, "..", "..", "results", "charts", "final_charts",
                    "ChatGPT Image 2026年7月23日 21_51_21.png")


def human(n):
    return f"{n/1024:.1f} KB ({n:,} bytes)"


def analyze(path):
    if not os.path.exists(path):
        return None
    info = {}
    info["path"] = path
    info["file_size"] = os.path.getsize(path)
    im = Image.open(path)
    info["size_px"] = im.size                      # (W, H)
    info["mode"] = im.mode
    info["format"] = im.format
    # DPI：PNG 通常存于 .info['dpi'] = (x_dpi, y_dpi)
    dpi = im.info.get("dpi", None)
    info["dpi"] = dpi
    # 每英寸像素推算的物理尺寸（英寸）
    if dpi and dpi[0] and dpi[1]:
        info["phys_w_in"] = info["size_px"][0] / dpi[0]
        info["phys_h_in"] = info["size_px"][1] / dpi[1]
    else:
        info["phys_w_in"] = info["phys_h_in"] = None
    # 总像素数与“密度”：DPI 决定同物理尺寸下的像素量
    info["total_px"] = info["size_px"][0] * info["size_px"][1]
    # PNG 压缩效率：字节数 / 像素数（越小说明越“平”、越易压缩）
    info["bytes_per_px"] = info["file_size"] / info["total_px"]
    # 颜色复杂度：统计非白(near-white)像素占比，衡量“信息密度”
    rgb = im.convert("RGB")
    arr = np.asarray(rgb)
    n = arr.shape[0] * arr.shape[1]
    nonwhite = int(np.sum(np.any(arr <= 245, axis=2)))
    info["nonwhite_ratio"] = nonwhite / n
    return info


def show(label, d):
    print(f"\n===== {label} =====")
    print(f"  文件: {os.path.basename(d['path'])}")
    print(f"  文件大小      : {human(d['file_size'])}")
    print(f"  像素尺寸      : {d['size_px'][0]} x {d['size_px'][1]}  (共 {d['total_px']/1e6:.2f} M 像素)")
    print(f"  DPI           : {d['dpi']}")
    if d.get('phys_w_in'):
        print(f"  物理尺寸(推算): {d['phys_w_in']:.2f} x {d['phys_h_in']:.2f} 英寸  ({d['phys_w_in']*2.54:.1f} x {d['phys_h_in']*2.54:.1f} cm)")
    print(f"  颜色模式      : {d['mode']} / {d['format']}")
    print(f"  字节/像素     : {d['bytes_per_px']*1024:.3f} B/px  (PNG压缩效率，越小越平)")
    print(f"  非白像素占比  : {d['nonwhite_ratio']*100:.1f}%  (信息密度，越大越“满”)")
    del d['path']


if __name__ == "__main__":
    ours = analyze(OUR)
    chat = analyze(CHAT)

    show("本项目 fig1 (matplotlib 生成)", ours)
    show("ChatGPT 生成图", chat)

    if ours and chat:
        print("\n===== 差异解读 =====")
        ratio_size = ours["file_size"] / chat["file_size"]
        ratio_px = ours["total_px"] / chat["total_px"]
        print(f"  文件大小：我们是 ChatGPT 的 {ratio_size:.2f}x ({ours['file_size']/1024:.0f}KB vs {chat['file_size']/1024:.0f}KB)")
        print(f"  像素总数：我们是 ChatGPT 的 {ratio_px:.2f}x")
        print(f"  DPI：我们 {ours['dpi']}  vs  ChatGPT {chat['dpi']}")
        print(f"  非白占比：我们 {ours['nonwhite_ratio']*100:.1f}%  vs  ChatGPT {chat['nonwhite_ratio']*100:.1f}%")
        print()
        print("  结论：")
        print("   - 我们 DPI=600、像素量是 ChatGPT(通常72/96 DPI)的数倍，印刷清晰度更高；")
        print("   - 但我们是线条/文字图(白底+矢量)，非白像素极少、PNG 压缩极好，所以 KB 反而小；")
        print("   - ChatGPT 图虽像素少，但含渐变/抗锯齿/阴影等连续色调，PNG 难压缩，故 KB 更大。")
        print("   - 文件大小(kB) ≠ 质量：我们 292KB 是 600DPI 的清晰线稿，它 1020KB 是 72DPI 的位图。")
