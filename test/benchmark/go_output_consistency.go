// go_output_consistency.go
// Go 输出一致性验证
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts（未设置线程参数）
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
//
// 测试目的：
// - 处理 bus.jpg 图像并进行推理
// - 提取 bounding boxes
// - 保存结果用于与 Python 版本比较
// - 确保输出一致性

package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// BoundingBox 表示检测到的边界框
type BoundingBox struct {
	X          float64
	Y          float64
	Width      float64
	Height     float64
	Confidence float64
	ClassID    int
}

// DetectionResult 表示检测结果
type DetectionResult struct {
	Boxes     []BoundingBox
	ModelName string
}

// ScaleInfo 缩放和填充信息
type ScaleInfo struct {
	ScaleX    float32
	ScaleY    float32
	PadLeft   int
	PadTop    int
	NewWidth  int
	NewHeight int
}

// 全局配置参数
var (
	confidenceThreshold = 0.25
	iouThreshold        = 0.7
	modelInputSize      = 640
	useRectScaling      = false
	stride              = 32

	// 内存池优化
	boundingBoxPool = sync.Pool{
		New: func() interface{} {
			return &boundingBox{}
		},
	}

	// 图像对象池
	imagePools     map[imageSizeKey]*sync.Pool
	imagePoolMutex sync.RWMutex
)

// imageSizeKey 用于标识不同尺寸的图像
type imageSizeKey struct {
	width  int
	height int
}

// boundingBox 表示检测到的目标的边界框
type boundingBox struct {
	label      string
	confidence float32
	x1, y1     float32
	x2, y2     float32
}

// GetImageFromPool 从图像池中获取指定尺寸的图像
func GetImageFromPool(width, height int) *image.RGBA {
	key := imageSizeKey{width: width, height: height}

	// 先尝试读取现有池
	imagePoolMutex.RLock()
	pool, exists := imagePools[key]
	imagePoolMutex.RUnlock()

	if !exists {
		// 如果池不存在，创建一个新池
		imagePoolMutex.Lock()
		// 再次检查，防止并发创建
		if pool, exists = imagePools[key]; !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return image.NewRGBA(image.Rect(0, 0, width, height))
				},
			}
			imagePools[key] = pool
		}
		imagePoolMutex.Unlock()
	}

	// 从池中获取图像
	img := pool.Get().(*image.RGBA)
	// 清空图像数据
	for i := range img.Pix {
		img.Pix[i] = 0
	}
	return img
}

// PutImageToPool 将图像归还到对应的尺寸池中
func PutImageToPool(img *image.RGBA) {
	if img == nil {
		return
	}

	bounds := img.Bounds()
	key := imageSizeKey{width: bounds.Dx(), height: bounds.Dy()}

	// 检查池是否存在
	imagePoolMutex.RLock()
	pool, exists := imagePools[key]
	imagePoolMutex.RUnlock()

	if exists {
		pool.Put(img)
	}
}

// min和max辅助函数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 标准 Letterbox 缩放
func resizeWithLetterbox(img image.Image, targetSize int) (image.Image, ScaleInfo) {
	bounds := img.Bounds()
	originalWidth, originalHeight := bounds.Dx(), bounds.Dy()

	// 官方逻辑：r = min(new_h / old_h, new_w / old_w)
	scale := math.Min(float64(targetSize)/float64(originalWidth), float64(targetSize)/float64(originalHeight))
	newWidth := int(math.Round(float64(originalWidth) * scale))
	newHeight := int(math.Round(float64(originalHeight) * scale))

	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)

	// 从对象池获取指定尺寸的图像
	result := GetImageFromPool(targetSize, targetSize)

	// 填充 114 灰色
	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			result.Set(x, y, color.RGBA{114, 114, 114, 255})
		}
	}

	// 居中计算：(total - new) / 2
	offsetX := (targetSize - newWidth) / 2
	offsetY := (targetSize - newHeight) / 2
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			result.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}

	return result, ScaleInfo{ScaleX: float32(scale), ScaleY: float32(scale), PadLeft: offsetX, PadTop: offsetY}
}

// Rect 缩放
func resizeWithRectScaling(img image.Image, targetSize int, stride int) (image.Image, ScaleInfo) {
	bounds := img.Bounds()
	originalWidth, originalHeight := bounds.Dx(), bounds.Dy()

	// 1. 计算缩放比例
	scale := math.Min(float64(targetSize)/float64(originalWidth), float64(targetSize)/float64(originalHeight))
	unpadWidth := int(math.Round(float64(originalWidth) * scale))
	unpadHeight := int(math.Round(float64(originalHeight) * scale))

	// 2. 官方核心逻辑：计算最小矩形填充 (dw, dh = np.mod(dw, stride))
	dw := targetSize - unpadWidth
	dh := targetSize - unpadHeight
	dw = dw % stride // 仅补充到能被 stride 整除
	dh = dh % stride

	// 3. 计算最终画布尺寸并居中
	finalWidth := unpadWidth + dw
	finalHeight := unpadHeight + dh

	resized := resize.Resize(uint(unpadWidth), uint(unpadHeight), img, resize.Bilinear)

	// 从对象池获取指定尺寸的图像
	result := GetImageFromPool(finalWidth, finalHeight)

	for y := 0; y < finalHeight; y++ {
		for x := 0; x < finalWidth; x++ {
			result.Set(x, y, color.RGBA{114, 114, 114, 255})
		}
	}

	offsetX, offsetY := dw/2, dh/2
	for y := 0; y < unpadHeight; y++ {
		for x := 0; x < unpadWidth; x++ {
			result.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}

	return result, ScaleInfo{ScaleX: float32(scale), ScaleY: float32(scale), PadLeft: offsetX, PadTop: offsetY}
}

// 确保值在指定范围内
func clamp(value, min, max float32) float32 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

// 准备输入数据
func prepareInput(pic image.Image, inputSize int) ([]float32, ScaleInfo, error) {
	channelSize := inputSize * inputSize
	data := make([]float32, 3*channelSize)
	var resizedImg image.Image
	var scaleInfo ScaleInfo
	if useRectScaling {
		resizedImg, scaleInfo = resizeWithRectScaling(pic, inputSize, stride)
	} else {
		resizedImg, scaleInfo = resizeWithLetterbox(pic, inputSize)
	}

	red := data[:channelSize]
	green := data[channelSize : 2*channelSize]
	blue := data[2*channelSize : 3*channelSize]

	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			r, g, b, _ := resizedImg.At(x, y).RGBA()
			idx := y*inputSize + x
			red[idx] = float32(r>>8) / 255.0
			green[idx] = float32(g>>8) / 255.0
			blue[idx] = float32(b>>8) / 255.0
		}
	}

	PutImageToPool(resizedImg.(*image.RGBA))
	return data, scaleInfo, nil
}

// 处理模型输出
func processOutput(output []float32, originalWidth, originalHeight int, confThreshold, iouThresh float32, scaleInfo ScaleInfo) []boundingBox {
	boundingBoxes := make([]*boundingBox, 0, 100)

	numAnchors := 8400
	numClasses := 80

	scaleX := scaleInfo.ScaleX
	scaleY := scaleInfo.ScaleY

	for idx := 0; idx < numAnchors; idx++ {
		// YOLO11: 前4维是 box (cx, cy, w, h)，后80维是类别置信度
		xc := output[0*numAnchors+idx]
		yc := output[1*numAnchors+idx]
		w := output[2*numAnchors+idx]
		h := output[3*numAnchors+idx]

		maxClsProb := float32(0)
		classID := 0
		for classIdx := 0; classIdx < numClasses; classIdx++ {
			clsProb := output[(4+classIdx)*numAnchors+idx]
			if clsProb > maxClsProb {
				maxClsProb = clsProb
				classID = classIdx
			}
		}

		finalConf := maxClsProb
		if finalConf < confThreshold {
			continue
		}

		// 映射回原图坐标
		origCenterX := (xc - float32(scaleInfo.PadLeft)) / scaleX
		origCenterY := (yc - float32(scaleInfo.PadTop)) / scaleY
		origW := w / scaleX
		origH := h / scaleY

		x1 := origCenterX - origW/2
		y1 := origCenterY - origH/2
		x2 := origCenterX + origW/2
		y2 := origCenterY + origH/2

		x1 = clamp(x1, 0, float32(originalWidth))
		y1 = clamp(y1, 0, float32(originalHeight))
		x2 = clamp(x2, 0, float32(originalWidth))
		y2 = clamp(y2, 0, float32(originalHeight))

		if x2 <= x1 || y2 <= y1 {
			continue
		}

		// 从对象池获取boundingBox
		box := boundingBoxPool.Get().(*boundingBox)
		box.label = yoloClasses[classID]
		box.confidence = finalConf
		box.x1 = x1
		box.y1 = y1
		box.x2 = x2
		box.y2 = y2
		boundingBoxes = append(boundingBoxes, box)
	}

	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	result := nonMaxSuppressionP(boundingBoxes, iouThresh)
	return result
}

// 非极大值抑制(NMS) - 指针版本
func nonMaxSuppressionP(boxes []*boundingBox, iouThreshold float32) []boundingBox {
	if len(boxes) == 0 {
		return []boundingBox{}
	}

	selected := make([]boundingBox, 0, len(boxes))
	picked := make([]bool, len(boxes))

	// 按类别分组进行NMS抑制
	for i := 0; i < len(boxes); i++ {
		if picked[i] {
			// 释放未选中的对象
			boundingBoxPool.Put(boxes[i])
			continue
		}

		// 保留选中的对象
		selected = append(selected, *boxes[i])
		picked[i] = true

		// 只对相同类别的框进行NMS抑制
		for j := i + 1; j < len(boxes); j++ {
			if picked[j] || boxes[i].label != boxes[j].label {
				continue
			}

			// 计算IoU
			iou := boxes[i].iou(boxes[j])
			if iou >= iouThreshold {
				picked[j] = true
				// 释放被抑制的对象
				boundingBoxPool.Put(boxes[j])
			}
		}
	}

	// 释放所有未处理的对象
	for i := 0; i < len(boxes); i++ {
		if !picked[i] {
			boundingBoxPool.Put(boxes[i])
		}
	}

	return selected
}

// iou 计算两个边界框的交并比
func (b *boundingBox) iou(other *boundingBox) float32 {
	intersection := b.intersection(other)
	union := b.area() + other.area() - intersection
	if union == 0 {
		return 0
	}
	return intersection / union
}

// area 计算边界框面积
func (b *boundingBox) area() float32 {
	return (b.x2 - b.x1) * (b.y2 - b.y1)
}

// intersection 计算两个边界框的交集面积
func (b *boundingBox) intersection(other *boundingBox) float32 {
	x1 := maxFloat32(b.x1, other.x1)
	y1 := maxFloat32(b.y1, other.y1)
	x2 := minFloat32(b.x2, other.x2)
	y2 := minFloat32(b.y2, other.y2)

	if x1 >= x2 || y1 >= y2 {
		return 0
	}

	return (x2 - x1) * (y2 - y1)
}

// 辅助函数
func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// YOLO类别标签
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// 安全的ONNX Runtime环境初始化函数
func initializeORTEnvironment(libPath string) error {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("初始化ORT环境失败: %w，使用的库路径: %s", err, libPath)
	}
	return nil
}

// 初始化ONNX Runtime会话
func initSession(modelPath string, inputSize int) (*ModelSession, error) {
	size := inputSize
	inputShape := ort.NewShape(1, 3, int64(size), int64(size))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %w", err)
	}
	outputShape := ort.NewShape(1, 84, 8400) // YOLO 输出
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出张量失败: %w", err)
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建SessionOptions失败: %w", err)
	}
	defer options.Destroy()
	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, options)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建ORT会话失败: %w", err)
	}
	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

// ModelSession 模型会话
type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

func (m *ModelSession) Destroy() {
	if m.Input != nil {
		m.Input.Destroy()
	}
	if m.Output != nil {
		m.Output.Destroy()
	}
	if m.Session != nil {
		m.Session.Destroy()
	}
}

// runInference 执行推理
func runInference(modelPath, imagePath, modelName, libPath string) (*DetectionResult, error) {
	fmt.Printf("\n===== Go 输出一致性测试 - %s ====\n", modelName)

	// 初始化ORT环境
	if err := initializeORTEnvironment(libPath); err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	// 初始化图像池
	imagePools = make(map[imageSizeKey]*sync.Pool)

	// 预处理图像
	fmt.Println("预处理图像...")
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("打开图像失败: %v", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("解码图像失败: %v", err)
	}

	originalWidth := img.Bounds().Dx()
	originalHeight := img.Bounds().Dy()

	// 初始化会话
	modelSession, err := initSession(modelPath, modelInputSize)
	if err != nil {
		return nil, err
	}
	defer modelSession.Destroy()

	// 准备输入
	inputData, scaleInfo, err := prepareInput(img, modelInputSize)
	if err != nil {
		return nil, err
	}

	// 复制数据到输入张量
	inputDataTensor := modelSession.Input.GetData()
	copy(inputDataTensor, inputData)

	// 执行推理
	fmt.Println("执行推理...")
	if err := modelSession.Session.Run(); err != nil {
		return nil, fmt.Errorf("推理失败: %v", err)
	}

	// 获取输出数据
	outputData := modelSession.Output.GetData()

	// 后处理输出
	fmt.Println("后处理输出...")
	boxes := processOutput(outputData, originalWidth, originalHeight, float32(confidenceThreshold), float32(iouThreshold), scaleInfo)

	// 转换为BoundingBox类型
	var boundingBoxes []BoundingBox
	for _, box := range boxes {
		classID := -1
		for i, class := range yoloClasses {
			if class == box.label {
				classID = i
				break
			}
		}
		if classID >= 0 {
			boundingBoxes = append(boundingBoxes, BoundingBox{
				X:          float64(box.x1),
				Y:          float64(box.y1),
				Width:      float64(box.x2 - box.x1),
				Height:     float64(box.y2 - box.y1),
				Confidence: float64(box.confidence),
				ClassID:    classID,
			})
		}
	}

	fmt.Printf("检测到 %d 个目标\n", len(boundingBoxes))
	for i, box := range boundingBoxes {
		fmt.Printf("目标 %d: 类别=%d, 置信度=%.4f, 坐标=(%.3f, %.3f, %.3f, %.3f)\n",
			i+1, box.ClassID, box.Confidence, box.X, box.Y, box.Width, box.Height)
	}

	return &DetectionResult{
		Boxes:     boundingBoxes,
		ModelName: modelName,
	}, nil
}

// saveDetectionResults 保存检测结果到文件
func saveDetectionResults(results []*DetectionResult, outputDir string) error {
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			return fmt.Errorf("创建输出目录失败: %v", err)
		}
	}

	for _, result := range results {
		outputPath := filepath.Join(outputDir, "go_"+result.ModelName+"_detections.txt")
		file, err := os.Create(outputPath)
		if err != nil {
			return fmt.Errorf("创建输出文件失败: %v", err)
		}

		_, err = fmt.Fprintf(file, "Model: %s\n", result.ModelName)
		if err != nil {
			file.Close()
			return fmt.Errorf("写入文件失败: %v", err)
		}

		_, err = fmt.Fprintf(file, "Number of detections: %d\n\n", len(result.Boxes))
		if err != nil {
			file.Close()
			return fmt.Errorf("写入文件失败: %v", err)
		}

		_, err = fmt.Fprintf(file, "Detections:\n")
		if err != nil {
			file.Close()
			return fmt.Errorf("写入文件失败: %v", err)
		}

		for i, box := range result.Boxes {
			_, err = fmt.Fprintf(file, "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%d\n",
				i+1, box.X, box.Y, box.Width, box.Height, box.Confidence, box.ClassID)
			if err != nil {
				file.Close()
				return fmt.Errorf("写入文件失败: %v", err)
			}
		}

		file.Close()
		fmt.Printf("检测结果已保存到: %s\n", outputPath)
	}

	return nil
}

func main() {
	fmt.Println("===== Go 输出一致性验证测试 =====")

	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前工作目录失败: %v\n", err)
		os.Exit(1)
	}

	// 构建模型路径
	modelPathLarge := filepath.Join(wd, "..", "..", "third_party", "yolo11x.onnx")
	modelPathSmall := filepath.Join(wd, "..", "..", "third_party", "yolo11n.onnx")

	// 构建项目根路径
	basePath := filepath.Join(wd, "..", "..")

	// 检查模型文件是否存在
	if !fileExists(modelPathLarge) {
		fmt.Printf("错误: 大模型文件不存在: %s\n", modelPathLarge)
		os.Exit(1)
	}
	if !fileExists(modelPathSmall) {
		fmt.Printf("错误: 轻模型文件不存在: %s\n", modelPathSmall)
		os.Exit(1)
	}

	// 构建图像路径
	imagePath := filepath.Join(basePath, "assets", "bus.jpg")
	if !fileExists(imagePath) {
		fmt.Printf("警告: 测试图像不存在: %s\n", imagePath)
		os.Exit(1)
	}

	// 构建库路径
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")

	// 执行推理
	var results []*DetectionResult

	result, err := runInference(modelPathLarge, imagePath, "yolo11x", libPath)
	if err != nil {
		fmt.Printf("大模型推理失败: %v\n", err)
		os.Exit(1)
	}
	results = append(results, result)

	result, err = runInference(modelPathSmall, imagePath, "yolo11n", libPath)
	if err != nil {
		fmt.Printf("轻模型推理失败: %v\n", err)
		os.Exit(1)
	}
	results = append(results, result)

	// 保存检测结果
	outputDir := filepath.Join(basePath, "results")
	if err := saveDetectionResults(results, outputDir); err != nil {
		fmt.Printf("保存检测结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n测试完成!")
}
