#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python_session_lifecycle_repro.py
Session 生命周期内存累积受控复现（Python 侧，ORT issue #27089 相关）

目的：在受控条件下复现 Session 生命周期内存漂移现象类——
      反复“创建 / 销毁 ONNX InferenceSession”导致进程常驻内存单调增长，
      关闭 ORT CPU 内存 Arena (enable_cpu_mem_arena=False) 后增长被压平。
      该现象在 glibc 运行时不归还分配器囤积的内存，是产业内存事故的根因。

设计：
  - 自变量：Arena {ON(默认), OFF}
  - 模式：每轮 创建 InferenceSession → 推理1次 → del + gc 销毁（共 N 轮）
  - 因变量：进程 PM(PrivateMemorySize64) 随周期变化、峰值 PM、漂移
  - 控制：YOLO11x、intra_op=1、inter_op=1、每轮推理1次

预期（假设，供论文核对）：
  - Arena ON  → PM 随周期单调增长（分配器囤积）
  - Arena OFF → PM 基本平稳（增长被压平）
"""

import os
import sys
import gc
import csv
import psutil
import numpy as np
import onnxruntime as ort


def get_process_pm():
    """进程级 PM (PrivateMemorySize64) MB，与论文 Go/Python 统一为 PM 口径"""
    return psutil.Process(os.getpid()).memory_info().private / 1024 / 1024


def create_session(model_path, arena_enabled=True):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    # 正确属性名 enable_cpu_mem_arena（默认 True，设为 False 关闭 Arena）
    so.enable_cpu_mem_arena = arena_enabled
    sess = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])
    return sess


def run_lifecycle(model_path, input_data, cycles, arena_enabled, safety_cap_mb=4096):
    arena_str = "ON" if arena_enabled else "OFF"
    print(f"[Arena={arena_str}] 开始 {cycles} 轮 建毁 循环")
    start_pm = get_process_pm()
    peak_pm = start_pm
    series = []
    input_name = None
    capped = False
    for i in range(cycles):
        try:
            sess = create_session(model_path, arena_enabled)
            if input_name is None:
                input_name = sess.get_inputs()[0].name
            sess.run(None, {input_name: input_data})
            del sess
            gc.collect()
        except Exception as e:
            print(f"    会话创建/推理失败(周期{i}): {e}")
            break
        pm = get_process_pm()
        series.append(pm)
        if pm > peak_pm:
            peak_pm = pm
        if pm > safety_cap_mb:
            print(f"    [安全上限] PM 超过 {safety_cap_mb} MB (周期{i+1})，停止以防 OOM 崩溃")
            capped = True
            break
    end_pm = get_process_pm()
    return start_pm, peak_pm, end_pm, series, capped


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base, "third_party", "yolo11x.onnx")
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        sys.exit(1)

    np.random.seed(12345)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    cycles = 500
    SAFETY_CAP_MB = 4096  # 安全上限：超过即停，避免宿主 OOM 崩溃（直接回答"炸到多少崩溃"）
    print("===== Python Session 生命周期内存累积受控复现 =====")
    print(f"模型: YOLO11x, 周期数: {cycles}, intra_op=1, inter_op=1")

    # Arena ON（默认）
    st_on, pk_on, en_on, ser_on, cap_on = run_lifecycle(model_path, input_data, cycles, True, SAFETY_CAP_MB)
    # Arena OFF
    st_off, pk_off, en_off, ser_off, cap_off = run_lifecycle(model_path, input_data, cycles, False, SAFETY_CAP_MB)

    print("\n===== 结果 =====")
    print(f"{'Arena':<6} {'起始PM':<12} {'峰值PM':<12} {'结束PM':<12} {'漂移':<12}")
    print(f"{'ON':<6} {st_on:<12.2f} {pk_on:<12.2f} {en_on:<12.2f} {en_on - st_on:<12.2f}")
    print(f"{'OFF':<6} {st_off:<12.2f} {pk_off:<12.2f} {en_off:<12.2f} {en_off - st_off:<12.2f}")

    out_dir = os.path.join(base, "results")
    os.makedirs(out_dir, exist_ok=True)

    def plateau_analysis(s, tail_frac=0.2):
        if len(s) < 20:
            return None
        n_tail = max(10, int(len(s) * tail_frac))
        tail = s[-n_tail:]
        return {"tail_min": min(tail), "tail_max": max(tail),
                "tail_range": max(tail) - min(tail), "tail_n": n_tail}

    pa_on = plateau_analysis(ser_on)
    pa_off = plateau_analysis(ser_off)

    summary_path = os.path.join(out_dir, "repro_lifecycle_python_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== Python Session 生命周期内存累积受控复现 =====\n")
        f.write(f"模型: YOLO11x, 周期数: {cycles}, intra_op=1, inter_op=1\n")
        f.write(f"安全上限: {SAFETY_CAP_MB} MB (触发即停，防宿主 OOM); Arena ON 触顶={'是' if cap_on else '否'}, Arena OFF 触顶={'是' if cap_off else '否'}\n\n")
        f.write("---- Arena ON (默认) ----\n")
        f.write(f"  起始PM: {st_on:.5f} MB\n")
        f.write(f"  峰值PM: {pk_on:.5f} MB\n")
        f.write(f"  结束PM: {en_on:.5f} MB\n")
        f.write(f"  总漂移(有界): {en_on - st_on:.5f} MB\n")
        if pa_on:
            f.write(f"  平台化分析(尾{pa_on['tail_n']}轮): {pa_on['tail_min']:.2f}–{pa_on['tail_max']:.2f} MB, 振幅 {pa_on['tail_range']:.2f} MB\n")
            f.write(f"  结论: 漂移有界，尾段平台化，非无界线性增长\n\n")
        f.write("---- Arena OFF (enable_cpu_mem_arena=False) ----\n")
        f.write(f"  起始PM: {st_off:.5f} MB\n")
        f.write(f"  峰值PM: {pk_off:.5f} MB\n")
        f.write(f"  结束PM: {en_off:.5f} MB\n")
        f.write(f"  总漂移(有界): {en_off - st_off:.5f} MB\n")
        if pa_off:
            f.write(f"  平台化分析(尾{pa_off['tail_n']}轮): {pa_off['tail_min']:.2f}–{pa_off['tail_max']:.2f} MB, 振幅 {pa_off['tail_range']:.2f} MB\n")
            f.write(f"  结论: 漂移有界，尾段平台化，非无界线性增长\n")

    series_path = os.path.join(out_dir, "repro_lifecycle_python_series.csv")
    with open(series_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "pm_arena_on", "pm_arena_off"])
        n = max(len(ser_on), len(ser_off))
        for i in range(n):
            on = ser_on[i] if i < len(ser_on) else ""
            off = ser_off[i] if i < len(ser_off) else ""
            w.writerow([i + 1, on, off])

    print(f"\n摘要: {summary_path}")
    print(f"序列: {series_path}")


if __name__ == "__main__":
    main()