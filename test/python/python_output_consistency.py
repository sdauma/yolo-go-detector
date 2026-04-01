# python_output_consistency.py
# Python 输出一致性验证
# 
# 测试目的：
# - 处理bus.jpg图像并进行推理
# - 提取bounding boxes
# - 保存结果用于与Go版本比较
# - 确保输出一致性

import onnxruntime as ort
import numpy as np
import os
import sys
from PIL import Image
from dataclasses import dataclass

# 固定随机种子，确保可复现
np.random.seed(12345)

# 全局配置参数
confidence_threshold = 0.25
iou_threshold = 0.7
model_input_size = 640
use_rect_scaling = False
stride = 32

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型路径 - 同时测试大模型和轻模型
model_path_large = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11x.onnx'))
model_path_small = os.path.abspath(os.path.join(current_dir, '..', '..', 'third_party', 'yolo11n.onnx'))

# 构建项目根路径
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 检查模型文件是否存在
if not os.path.exists(model_path_large):
    print(f"错误: 大模型文件不存在: {model_path_large}")
    sys.exit(1)
if not os.path.exists(model_path_small):
    print(f"错误: 轻模型文件不存在: {model_path_small}")
    sys.exit(1)

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    
    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.width = float(self.width)
        self.height = float(self.height)
        self.confidence = float(self.confidence)
        self.class_id = int(self.class_id)

@dataclass
class DetectionResult:
    boxes: list
    model_name: str

@dataclass
class ScaleInfo:
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    new_width: int
    new_height: int

def resize_with_letterbox(image, target_size):
    """使用letterbox方法调整图像大小"""
    original_width, original_height = image.size
    
    # 计算缩放比例
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    
    # 调整大小
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # 创建目标大小的画布
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    
    # 计算填充位置
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    
    # 将调整后的图像绘制到画布中心
    canvas.paste(resized, (pad_left, pad_top))
    
    return canvas, ScaleInfo(
        scale_x=scale, scale_y=scale, pad_left=pad_left, pad_top=pad_top,
        new_width=new_width, new_height=new_height
    )

def resize_with_rect_scaling(image, target_size, stride):
    """使用rect缩放方法调整图像大小"""
    original_width, original_height = image.size
    
    # 计算缩放比例
    scale = min(target_size / original_width, target_size / original_height)
    unpad_width = int(round(original_width * scale))
    unpad_height = int(round(original_height * scale))
    
    # 计算最小矩形填充
    dw = target_size - unpad_width
    dh = target_size - unpad_height
    dw = dw % stride
    dh = dh % stride
    
    # 计算最终画布尺寸
    final_width = unpad_width + dw
    final_height = unpad_height + dh
    
    # 调整大小
    resized = image.resize((unpad_width, unpad_height), Image.Resampling.BILINEAR)
    
    # 创建目标大小的画布
    canvas = Image.new('RGB', (final_width, final_height), (114, 114, 114))
    
    # 计算填充位置
    pad_left = dw // 2
    pad_top = dh // 2
    
    # 将调整后的图像绘制到画布中心
    canvas.paste(resized, (pad_left, pad_top))
    
    return canvas, ScaleInfo(
        scale_x=scale, scale_y=scale, pad_left=pad_left, pad_top=pad_top,
        new_width=unpad_width, new_height=unpad_height
    )

def preprocess_image(image_path, input_size=640):
    """预处理图像，调整大小并归一化"""
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # 调整大小
    if use_rect_scaling:
        resized_img, scale_info = resize_with_rect_scaling(image, input_size, stride)
    else:
        resized_img, scale_info = resize_with_letterbox(image, input_size)
    
    # 转换为numpy数组
    image_array = np.array(resized_img)
    
    # 归一化
    image_array = image_array.astype(np.float32) / 255.0
    
    # 调整维度顺序 (H, W, C) -> (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 添加批次维度
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array, original_width, original_height, scale_info

def non_max_suppression(boxes, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    # 按置信度降序排序
    boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
    
    selected_boxes = []
    picked = [False] * len(boxes)
    
    # 按类别分组进行NMS
    for i in range(len(boxes)):
        if picked[i]:
            continue
        
        selected_boxes.append(boxes[i])
        picked[i] = True
        
        # 只对相同类别的框进行NMS
        for j in range(i + 1, len(boxes)):
            if picked[j] or boxes[i].class_id != boxes[j].class_id:
                continue
            
            # 计算IoU
            iou = calculate_iou(boxes[i], boxes[j])
            if iou >= iou_threshold:
                picked[j] = True
    
    return selected_boxes

def calculate_iou(box1, box2):
    """计算两个边界框的交并比"""
    # 计算交集
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)
    
    if x1 >= x2 or y1 >= y2:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union

def postprocess_output(output, original_width, original_height, scale_info, confidence_threshold=0.25, iou_threshold=0.7):
    """后处理模型输出，提取bounding boxes"""
    # YOLO11输出格式: (1, 84, 8400) -> (batch, 84, num_predictions)
    # 84 = 4 (bounding box) + 80 (classes)
    
    boxes = []
    
    # 获取预测结果
    predictions = output[0]
    

    
    num_anchors = 8400
    num_classes = 80
    
    scale_x = scale_info.scale_x
    scale_y = scale_info.scale_y
    pad_left = scale_info.pad_left
    pad_top = scale_info.pad_top
    
    # 遍历所有预测
    for idx in range(num_anchors):
        # 获取bounding box坐标（中心点坐标和宽高）
        xc = predictions[0, 0, idx]
        yc = predictions[0, 1, idx]
        w = predictions[0, 2, idx]
        h = predictions[0, 3, idx]
        
        # 获取最大类别置信度
        cls_probs = predictions[0, 4:, idx]
        max_cls_prob = np.max(cls_probs)
        class_id = np.argmax(cls_probs)
        
        final_conf = max_cls_prob
        if final_conf < confidence_threshold:
            continue
        
        # 映射回原图坐标
        orig_center_x = (xc - pad_left) / scale_x
        orig_center_y = (yc - pad_top) / scale_y
        orig_w = w / scale_x
        orig_h = h / scale_y
        
        # 转换为左上角坐标
        x1 = orig_center_x - orig_w / 2
        y1 = orig_center_y - orig_h / 2
        x2 = orig_center_x + orig_w / 2
        y2 = orig_center_y + orig_h / 2
        
        # 限制在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width, x2)
        y2 = min(original_height, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        boxes.append(BoundingBox(
            x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1),
            confidence=float(final_conf), class_id=int(class_id)
        ))
    
    # 执行非极大值抑制
    boxes = non_max_suppression(boxes, iou_threshold)
    
    return boxes

def run_inference(model_path, image_path, model_name):
    print(f"\n===== Python 输出一致性测试 - {model_name} ====")
    
    # 预处理图像
    print("预处理图像...")
    input_data, original_width, original_height, scale_info = preprocess_image(image_path, model_input_size)
    
    # 创建 Session
    print("创建 InferenceSession...")
    try:
        sess_options = ort.SessionOptions()
        
        # 显式设置所有 SessionOptions 参数
        # 线程配置 - 12线程，与其他测试保持一致
        sess_options.intra_op_num_threads = 12
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"错误: 创建 InferenceSession 失败: {e}")
        sys.exit(1)

    # 获取输入信息
    input_name = sess.get_inputs()[0].name

    # 执行推理
    print("执行推理...")
    output = sess.run(None, {input_name: input_data})
    
    # 后处理输出
    print("后处理输出...")
    boxes = postprocess_output(output, original_width, original_height, scale_info, confidence_threshold, iou_threshold)
    
    print(f"检测到 {len(boxes)} 个目标")
    for i, box in enumerate(boxes):
        print(f"目标 {i+1}: 类别={int(box.class_id)}, 置信度={float(box.confidence):.4f}, 坐标=({float(box.x):.3f}, {float(box.y):.3f}, {float(box.width):.3f}, {float(box.height):.3f})")
    
    return DetectionResult(boxes=boxes, model_name=model_name)

def save_detection_results(results, output_dir):
    """保存检测结果到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for result in results:
        output_path = os.path.join(output_dir, f"python_{result.model_name}_detections.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {result.model_name}\n")
            f.write(f"Number of detections: {len(result.boxes)}\n")
            f.write("\nDetections:\n")
            for i, box in enumerate(result.boxes):
                f.write(f"{i+1},{box.x:.5f},{box.y:.5f},{box.width:.5f},{box.height:.5f},{box.confidence:.5f},{box.class_id}\n")
        print(f"检测结果已保存到: {output_path}")

def main():
    print("===== Python 输出一致性验证测试 =====")

    # 测试图像路径
    image_path = os.path.join(base_path, "assets", "bus.jpg")
    if not os.path.exists(image_path):
        print(f"警告: 测试图像不存在: {image_path}")
        print("跳过输出一致性验证测试")
        return

    # 运行测试 - 大模型和轻模型
    results = []
    results.append(run_inference(model_path_large, image_path, "yolo11x"))
    results.append(run_inference(model_path_small, image_path, "yolo11n"))

    # 保存检测结果
    output_dir = os.path.join(base_path, "results")
    save_detection_results(results, output_dir)

    print("\n测试完成!")

if __name__ == "__main__":
    main()
