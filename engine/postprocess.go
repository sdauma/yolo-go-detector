package engine

import (
	"sort"
)

// BoundingBox 边界框
type BoundingBox struct {
	XMin      float32
	YMin      float32
	XMax      float32
	YMax      float32
	ClassID   int
	Confidence float32
}

// NMSConfig NMS 配置
type NMSConfig struct {
	ConfThreshold float32 // 置信度阈值
	IoUThreshold  float32 // IoU 阈值
	MaxDetections int     // 最大检测数量
}

// DefaultNMSConfig 默认 NMS 配置
func DefaultNMSConfig() NMSConfig {
	return NMSConfig{
		ConfThreshold: 0.25,
		IoUThreshold:  0.45,
		MaxDetections: 300,
	}
}

// Postprocessor 后处理器
type Postprocessor struct {
	config NMSConfig
}

// NewPostprocessor 创建后处理器
func NewPostprocessor(config NMSConfig) *Postprocessor {
	return &Postprocessor{
		config: config,
	}
}

// Process 处理 YOLO 输出，执行 NMS
func (p *Postprocessor) Process(output []float32, inputWidth, inputHeight int) []BoundingBox {
	// YOLO11x 输出格式: [1, 84, 8400]
	// 84 = 4 (bbox) + 80 (classes)
	// 8400 = number of anchors

	numAnchors := 8400
	numClasses := 80

	detections := make([]BoundingBox, 0, numAnchors)

	// 解析输出
	// YOLO 输出内存布局为行主序 [84, 8400]（与主项目 main.go 完全一致）
	// 前 4 行是 box (cx, cy, w, h)，后 80 行是类别置信度
	for i := 0; i < numAnchors; i++ {
		// 获取边界框坐标 (cx, cy, w, h) — 行主序
		cx := output[0*numAnchors+i]
		cy := output[1*numAnchors+i]
		w := output[2*numAnchors+i]
		h := output[3*numAnchors+i]

		// 转换为 (x1, y1, x2, y2) — 仍在模型输入空间
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2

		// 获取最大置信度和对应的类别
		maxConf := float32(0)
		classID := 0
		for c := 0; c < numClasses; c++ {
			conf := output[(4+c)*numAnchors+i]
			if conf > maxConf {
				maxConf = conf
				classID = c
			}
		}

		// 过滤低置信度检测
		if maxConf < p.config.ConfThreshold {
			continue
		}

		detections = append(detections, BoundingBox{
			XMin:      x1,
			YMin:      y1,
			XMax:      x2,
			YMax:      y2,
			ClassID:   classID,
			Confidence: maxConf,
		})
	}

	// 按置信度降序排序
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})

	// 执行 NMS
	return p.nms(detections)
}

// nms 执行非极大值抑制
func (p *Postprocessor) nms(detections []BoundingBox) []BoundingBox {
	if len(detections) == 0 {
		return detections
	}

	selected := make([]BoundingBox, 0, len(detections))
	suppressed := make([]bool, len(detections))

	for i := 0; i < len(detections); i++ {
		if suppressed[i] {
			continue
		}

		selected = append(selected, detections[i])

		if len(selected) >= p.config.MaxDetections {
			break
		}

		// 抑制重叠的检测框
		for j := i + 1; j < len(detections); j++ {
			if suppressed[j] {
				continue
			}

			// 只抑制同一类别的检测
			if detections[i].ClassID != detections[j].ClassID {
				continue
			}

			iou := calculateIoU(detections[i], detections[j])
			if iou > p.config.IoUThreshold {
				suppressed[j] = true
			}
		}
	}

	return selected
}

// calculateIoU 计算 IoU (Intersection over Union)
func calculateIoU(box1, box2 BoundingBox) float32 {
	// 计算交集
	x1 := maxFloat32(box1.XMin, box2.XMin)
	y1 := maxFloat32(box1.YMin, box2.YMin)
	x2 := minFloat32(box1.XMax, box2.XMax)
	y2 := minFloat32(box1.YMax, box2.YMax)

	intersection := maxFloat32(0, x2-x1) * maxFloat32(0, y2-y1)

	// 计算并集
	area1 := (box1.XMax - box1.XMin) * (box1.YMax - box1.YMin)
	area2 := (box2.XMax - box2.XMin) * (box2.YMax - box2.YMin)
	union := area1 + area2 - intersection

	if union == 0 {
		return 0
	}

	return intersection / union
}

// maxFloat32 返回两个 float32 的最大值
func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// minFloat32 返回两个 float32 的最小值
func minFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// ScaleBboxes 缩放边界框到原始图像尺寸
func (p *Postprocessor) ScaleBboxes(boxes []BoundingBox, inputWidth, inputHeight, originalWidth, originalHeight int) []BoundingBox {
	scaled := make([]BoundingBox, len(boxes))

	scaleX := float32(originalWidth) / float32(inputWidth)
	scaleY := float32(originalHeight) / float32(inputHeight)

	for i, box := range boxes {
		scaled[i] = BoundingBox{
			XMin:      box.XMin * scaleX,
			YMin:      box.YMin * scaleY,
			XMax:      box.XMax * scaleX,
			YMax:      box.YMax * scaleY,
			ClassID:   box.ClassID,
			Confidence: box.Confidence,
		}
	}

	return scaled
}

// GetClassNames 获取类别名称（COCO 数据集）
func GetClassNames() []string {
	return []string{
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
		"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
		"teddy bear", "hair drier", "toothbrush",
	}
}

// GetClassName 根据类别 ID 获取类别名称
func GetClassName(classID int) string {
	names := GetClassNames()
	if classID >= 0 && classID < len(names) {
		return names[classID]
	}
	return "unknown"
}

// DetectionInfo 检测信息
type DetectionInfo struct {
	ClassID   int
	ClassName string
	Confidence float32
	XMin      float32
	YMin      float32
	XMax      float32
	YMax      float32
}

// ConvertToDetectionInfo 转换为检测信息格式
func ConvertToDetectionInfo(boxes []BoundingBox) []DetectionInfo {
	info := make([]DetectionInfo, len(boxes))
	for i, box := range boxes {
		info[i] = DetectionInfo{
			ClassID:   box.ClassID,
			ClassName: GetClassName(box.ClassID),
			Confidence: box.Confidence,
			XMin:      box.XMin,
			YMin:      box.YMin,
			XMax:      box.XMax,
			YMax:      box.YMax,
		}
	}
	return info
}
