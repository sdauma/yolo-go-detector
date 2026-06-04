"""
test/compare/compare.py
Python vs Go YOLO 检测结果一致性对比
用法: python compare.py
"""
import subprocess, sys, os, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
COMPARE_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(PROJECT_ROOT, "third_party")


def run_python_detection():
    """运行 Python YOLO 检测"""
    img = os.path.join(ASSETS_DIR, "bus.jpg")
    model = os.path.join(THIRD_PARTY, "yolo11x.onnx")

    cmd = [
        "yolo", "predict", "task=detect",
        f"model={model}", "imgsz=640",
        f"source={img}", "conf=0.25", "iou=0.7",
        "save_txt=true", "save_conf=true",
        f"project={COMPARE_DIR}", "name=py_result", "exist_ok=true"
    ]
    print(f"\n{'='*60}")
    print("运行 Python YOLO 检测...")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=PROJECT_ROOT)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Python 检测失败:\n{result.stderr}")
        return None
    # PyTorch 模式路径: predict/labels/, ONNX 模式路径: labels/
    path1 = os.path.join(COMPARE_DIR, "py_result", "predict", "labels", "bus.txt")
    path2 = os.path.join(COMPARE_DIR, "py_result", "labels", "bus.txt")
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    print(f"警告: 找不到 Python 标注文件")
    return None


def run_go_detection():
    """运行 Go 检测"""
    go_exe = os.path.join(COMPARE_DIR, "compare.exe")
    if not os.path.exists(go_exe):
        print(f"错误: 找不到 compare.exe, 请先编译: go build -o test\\compare\\compare.exe .\\test\\compare\\")
        return None

    print(f"\n{'='*60}")
    print("运行 Go 检测...")
    print(f"{'='*60}")

    result = subprocess.run(
        [go_exe, "-img", os.path.join(ASSETS_DIR, "bus.jpg")],
        capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=COMPARE_DIR
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Go 检测失败:\n{result.stderr}")
        return None
    return os.path.join(COMPARE_DIR, "bus_go_detections.txt")


def parse_yolo_txt(txt_path):
    """解析 YOLO 格式 txt 文件 (class cx cy w h conf) — 归一化坐标，自动去重"""
    detections = []
    seen = set()
    if not os.path.exists(txt_path):
        print(f"  警告: 找不到文件 {txt_path}")
        return detections
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                # 去重：四舍五入到 6 位小数后判断
                key = (cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
                if key not in seen:
                    seen.add(key)
                    detections.append((cls, cx, cy, w, h, conf))
    return detections


def iou(box1, box2):
    """计算两个框 (cx,cy,w,h) 的 IoU"""
    x1_1, y1_1 = box1[1] - box1[3]/2, box1[2] - box1[4]/2
    x2_1, y2_1 = box1[1] + box1[3]/2, box1[2] + box1[4]/2
    x1_2, y1_2 = box2[1] - box2[3]/2, box2[2] - box2[4]/2
    x2_2, y2_2 = box2[1] + box2[3]/2, box2[2] + box2[4]/2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compare(py_dets, go_dets):
    """对比两组检测结果"""
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    print(f"  Python 检测: {len(py_dets)} 个目标")
    print(f"  Go 检测:     {len(go_dets)} 个目标")

    coco_names = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ]

    print("\nPython 检测详情:")
    for i, d in enumerate(py_dets):
        cls, cx, cy, w, h, conf = d
        name = coco_names[cls] if cls < len(coco_names) else f"class_{cls}"
        print(f"  [{i+1}] {name}(id={cls}) cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f} conf={conf:.4f}")

    print("\nGo 检测详情:")
    for i, d in enumerate(go_dets):
        cls, cx, cy, w, h, conf = d
        name = coco_names[cls] if cls < len(coco_names) else f"class_{cls}"
        print(f"  [{i+1}] {name}(id={cls}) cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f} conf={conf:.4f}")

    # 匹配检测
    print("\n匹配分析:")
    matched_py = set()
    matched_go = set()
    total_iou = 0.0
    match_count = 0

    for pi, pd in enumerate(py_dets):
        best_iou = 0
        best_gi = -1
        for gi, gd in enumerate(go_dets):
            if gi in matched_go:
                continue
            if pd[0] != gd[0]:  # 同类别
                continue
            i = iou(pd, gd)
            if i > best_iou:
                best_iou = i
                best_gi = gi
        if best_iou >= 0.5 and best_gi >= 0:
            matched_py.add(pi)
            matched_go.add(best_gi)
            total_iou += best_iou
            match_count += 1
            name = coco_names[pd[0]] if pd[0] < len(coco_names) else f"class_{pd[0]}"
            print(f"  ✅ {name}: Python[{pi+1}] ↔ Go[{best_gi+1}] IoU={best_iou:.4f}")

    unmatched_py = len(py_dets) - len(matched_py)
    unmatched_go = len(go_dets) - len(matched_go)

    if unmatched_py > 0:
        for pi in range(len(py_dets)):
            if pi not in matched_py:
                name = coco_names[py_dets[pi][0]] if py_dets[pi][0] < len(coco_names) else f"class_{py_dets[pi][0]}"
                print(f"  ❌ Python 独有: [{pi+1}] {name}")
    if unmatched_go > 0:
        for gi in range(len(go_dets)):
            if gi not in matched_go:
                name = coco_names[go_dets[gi][0]] if go_dets[gi][0] < len(coco_names) else f"class_{go_dets[gi][0]}"
                print(f"  ❌ Go 独有: [{gi+1}] {name}")

    avg_iou = total_iou / match_count if match_count > 0 else 0

    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"  Python 检测数:   {len(py_dets)}")
    print(f"  Go 检测数:       {len(go_dets)}")
    print(f"  匹配数 (IoU≥0.5): {match_count}")
    print(f"  Python 独有:     {unmatched_py}")
    print(f"  Go 独有:         {unmatched_go}")
    print(f"  匹配平均 IoU:    {avg_iou:.4f}")

    if len(py_dets) == len(go_dets) and unmatched_py == 0 and unmatched_go == 0 and avg_iou > 0.9:
        print("\n  🎉 结论: 完全一致！")
    elif match_count > 0 and unmatched_py <= 1 and unmatched_go <= 1:
        print(f"\n  ✅ 结论: 基本一致（{match_count}个匹配，微小区间差异）")
    else:
        print(f"\n  ⚠️ 结论: 存在差异，需检查")


def main():
    py_txt = run_python_detection()
    go_txt = run_go_detection()

    if py_txt is None or go_txt is None:
        print("\n 错误: 检测未完成，无法对比")
        sys.exit(1)

    py_dets = parse_yolo_txt(py_txt)
    go_dets = parse_yolo_txt(go_txt)

    compare(py_dets, go_dets)


if __name__ == "__main__":
    main()
