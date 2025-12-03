package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/8ff/prettyTimer"
	"github.com/flopp/go-findfont" // 添加字体查找库
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/font"
	"golang.org/x/image/font/inconsolata" // 用于回退的默认字体
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// 全局配置
var (
	modelPath       = "./third_party/yolo11n.onnx"
	imagePath       = "bus.jpg"
	outputImagePath = "bus-detected.jpg"
	useCoreML       = false

	//高召回要求（如安防） 降低 conf（0.2~0.3），提高 iou（0.6~0.7）
	confidenceThreshold = flag.Float64("conf", 0.25, "置信度阈值")
	iouThreshold        = flag.Float64("iou", 0.7, "IOU阈值")
	modelInputSize      = flag.Int("size", 640, "模型输入尺寸")

	// 中文字体变量
	chineseFont font.Face

	// ONNX Runtime 初始化状态控制
	ortInitialized bool
	ortInitMutex   sync.Mutex
)

func main() {
	// 设置环境变量确保UTF-8编码支持
	os.Setenv("LC_ALL", "zh_CN.UTF-8")

	flag.Parse()
	fmt.Printf("使用参数: conf=%.2f, iou=%.2f, size=%d\n", *confidenceThreshold, *iouThreshold, *modelInputSize)
	os.Exit(run())

}

// initChineseFont 初始化中文字体
func initChineseFont() error {
	fontPaths := findfont.List()
	var fontPath string

	preferredFonts := []string{
		"simhei.ttf",
		"simkai.ttf",
		"simfang.ttf",
		"SIMLI.TTF",
		"msyh.ttf",
		"msyhbd.ttf",
		"simsun.ttc",
		"Deng.ttf",
	}

	for _, preferredFont := range preferredFonts {
		for _, path := range fontPaths {
			if strings.Contains(strings.ToLower(path), strings.ToLower(preferredFont)) {
				fontPath = path
				break
			}
		}
		if fontPath != "" {
			break
		}
	}

	if fontPath == "" {
		return fmt.Errorf("未找到可用的中文字体")
	}

	fontData, err := os.ReadFile(fontPath)
	if err != nil {
		return fmt.Errorf("读取字体文件失败: %w", err)
	}

	fontTT, err := opentype.Parse(fontData)
	if err != nil {
		return fmt.Errorf("解析字体失败: %w", err)
	}

	chineseFont, err = opentype.NewFace(fontTT, &opentype.FaceOptions{
		Size:    20,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return fmt.Errorf("创建字体face失败: %w", err)
	}

	fmt.Printf("已加载中文字体: %s\n", fontPath)
	return nil
}

// cleanupFont 清理字体资源
func cleanupFont() {
	if chineseFont != nil {
		chineseFont.Close()
	}
}

// getChineseLabel 获取中文标签
func getChineseLabel(englishLabel string) string {
	if chinese, exists := objectMap[englishLabel]; exists {
		return chinese
	}
	return englishLabel
}

func run() int {
	// 初始化中文字体 [6](@ref)
	if err := initChineseFont(); err != nil {
		fmt.Printf("警告: 中文字体初始化失败: %v, 将尝试使用默认字体\n", err)
	} else {
		defer cleanupFont() // 确保程序退出时清理字体资源
	}

	timingStats := prettyTimer.NewTimingStats()

	if os.Getenv("USE_COREML") == "true" {
		useCoreML = true
	}

	originalPic, e := loadImageFile(imagePath)
	if e != nil {
		fmt.Printf("加载输入图像错误: %s\n", e)
		return 1
	}
	originalWidth := originalPic.Bounds().Dx()
	originalHeight := originalPic.Bounds().Dy()
	fmt.Printf("已加载图像: %dx%d\n", originalWidth, originalHeight)

	modelSession, e := initSession()
	if e != nil {
		fmt.Printf("创建会话和张量错误: %s\n", e)
		return 1
	}
	defer modelSession.Destroy()

	e = prepareInput(originalPic, modelSession.Input)
	if e != nil {
		fmt.Printf("转换图像到网络输入错误: %s\n", e)
		return 1
	}

	timingStats.Start()
	e = modelSession.Session.Run()
	if e != nil {
		fmt.Printf("运行ORT会话错误: %s\n", e)
		return 1
	}
	timingStats.Finish()

	allBoxes := processOutput(modelSession.Output.GetData(), originalWidth, originalHeight,
		float32(*confidenceThreshold), float32(*iouThreshold))

	fmt.Printf("检测到 %d 个对象:\n", len(allBoxes))
	for i, box := range allBoxes {
		chineseLabel := getChineseLabel(box.label)
		fmt.Printf("框 %d: %s (置信度: %.4f): (%.1f, %.1f, %.1f, %.1f)\n",
			i, chineseLabel, box.confidence, box.x1, box.y1, box.x2, box.y2)
	}

	// 确保目录存在
	if err := ensureDirForFile(outputImagePath); err != nil {
		fmt.Printf("输出图像已保存至: %s\n", err.Error())
	}

	e = drawBoundingBoxesWithLabels(originalPic, allBoxes, outputImagePath)
	if e != nil {
		fmt.Printf("绘制边界框错误: %s\n", e)
		return 1
	}
	fmt.Printf("输出图像已保存至: %s\n", outputImagePath)

	timingStats.PrintStats()
	return 0
}

// ensureDirForFile 确保给定文件路径的父目录存在，若不存在则创建
func ensureDirForFile(filePath string) error {
	// 获取文件所在目录（即去掉文件名）
	dir := filepath.Dir(filePath)

	// 检查目录是否存在
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		// 递归创建所有缺失的目录
		err = os.MkdirAll(dir, 0755) // 权限: rwxr-xr-x
		if err != nil {
			return fmt.Errorf("无法创建目录 %s: %w", dir, err)
		}
		fmt.Printf("✅ 目录已创建: %s\n", dir)
	} else if err != nil {
		return fmt.Errorf("检查目录状态失败 %s: %w", dir, err)
	} else {
		fmt.Printf("📁 目录已存在: %s\n", dir)
	}
	return nil
}

// ModelSession 封装ONNX运行时会话和张量
type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

func (m *ModelSession) Destroy() {
	m.Session.Destroy()
	m.Input.Destroy()
	m.Output.Destroy()
}

// boundingBox 表示检测到的目标的边界框
type boundingBox struct {
	label      string
	confidence float32
	x1, y1     float32
	x2, y2     float32
}

func (b *boundingBox) String() string {
	chineseLabel := getChineseLabel(b.label)
	return fmt.Sprintf("对象 %s (置信度 %.4f): (%.1f, %.1f, %.1f, %.1f)",
		chineseLabel, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

func (b *boundingBox) toRect() image.Rectangle {
	return image.Rect(int(b.x1+0.5), int(b.y1+0.5), int(b.x2+0.5), int(b.y2+0.5))
}

func (b *boundingBox) area() float32 {
	w := b.x2 - b.x1
	h := b.y2 - b.y1
	return w * h
}

func (b *boundingBox) intersection(other *boundingBox) float32 {
	r1 := b.toRect()
	r2 := other.toRect()
	intersected := r1.Intersect(r2).Size()
	return float32(intersected.X * intersected.Y)
}

func (b *boundingBox) union(other *boundingBox) float32 {
	intersectArea := b.intersection(other)
	totalArea := b.area() + other.area()
	return totalArea - intersectArea
}

func (b *boundingBox) iou(other *boundingBox) float32 {
	return b.intersection(other) / b.union(other)
}

func loadImageFile(filePath string) (image.Image, error) {
	f, e := os.Open(filePath)
	if e != nil {
		return nil, fmt.Errorf("打开 %s 错误: %w", filePath, e)
	}
	defer f.Close()
	pic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("解码 %s 错误: %w", filePath, e)
	}
	return pic, nil
}

func prepareInput(pic image.Image, dst *ort.Tensor[float32]) error {
	data := dst.GetData()
	inputSize := *modelInputSize
	channelSize := inputSize * inputSize

	if len(data) < (channelSize * 3) {
		return fmt.Errorf("目标张量仅包含 %d 个浮点数，需要 %d", len(data), channelSize*3)
	}

	redChannel := data[0:channelSize]
	greenChannel := data[channelSize : channelSize*2]
	blueChannel := data[channelSize*2 : channelSize*3]

	resizedImg := resizeWithAspectRatio(pic, inputSize, inputSize)

	i := 0
	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			r, g, b, _ := resizedImg.At(x, y).RGBA()
			redChannel[i] = float32(r>>8) / 255.0
			greenChannel[i] = float32(g>>8) / 255.0
			blueChannel[i] = float32(b>>8) / 255.0
			i++
		}
	}
	return nil
}

func resizeWithAspectRatio(img image.Image, targetWidth, targetHeight int) image.Image {
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	scale := float64(targetWidth) / float64(originalWidth)
	if float64(targetHeight)/float64(originalHeight) < scale {
		scale = float64(targetHeight) / float64(originalHeight)
	}

	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)
	result := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))

	draw.Draw(result, result.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)

	offsetX := (targetWidth - newWidth) / 2
	offsetY := (targetHeight - newHeight) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+newWidth, offsetY+newHeight),
		resized, image.Point{}, draw.Src)

	return result
}

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	panic("无法找到支持此系统的onnxruntime库版本")
}

func initializeORTEnvironment() error {
	ortInitMutex.Lock()
	defer ortInitMutex.Unlock()

	if ortInitialized {
		return nil // 已经初始化，直接返回
	}

	ort.SetSharedLibraryPath(getSharedLibPath())
	err := ort.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("初始化ORT环境错误: %w", err)
	}

	ortInitialized = true
	//fmt.Println("ONNX Runtime环境初始化成功")
	return nil
}

func initSession() (*ModelSession, error) {

	// 先初始化ONNX Runtime环境
	if err := initializeORTEnvironment(); err != nil {
		return nil, err
	}

	inputSize := *modelInputSize
	inputShape := ort.NewShape(1, 3, int64(inputSize), int64(inputSize))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量错误: %w", err)
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出张量错误: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建ORT会话选项错误: %w", err)
	}
	defer options.Destroy()

	if useCoreML {
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			inputTensor.Destroy()
			outputTensor.Destroy()
			return nil, fmt.Errorf("启用CoreML错误: %w", err)
		}
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建ORT会话错误: %w", err)
	}

	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

func processOutput(output []float32, originalWidth, originalHeight int, confThreshold, iouThresh float32) []boundingBox {
	boundingBoxes := make([]boundingBox, 0, 8400)

	scaleUsed := float32(640.0) / float32(originalWidth)
	scaleH := float32(640.0) / float32(originalHeight)
	if scaleH < scaleUsed {
		scaleUsed = scaleH
	}

	newWidth := int(float32(originalWidth) * scaleUsed)
	newHeight := int(float32(originalHeight) * scaleUsed)
	padX := (640 - newWidth) / 2
	padY := (640 - newHeight) / 2

	for idx := 0; idx < 8400; idx++ {
		xc := output[idx]
		yc := output[8400+idx]
		w := output[2*8400+idx]
		h := output[3*8400+idx]

		maxProb := float32(0)
		classID := 0
		for classIdx := 0; classIdx < 80; classIdx++ {
			prob := output[8400*(classIdx+4)+idx]
			if prob > maxProb {
				maxProb = prob
				classID = classIdx
			}
		}

		if maxProb < confThreshold {
			continue
		}

		x1 := ((xc - w/2) - float32(padX)) / scaleUsed
		y1 := ((yc - h/2) - float32(padY)) / scaleUsed
		x2 := ((xc + w/2) - float32(padX)) / scaleUsed
		y2 := ((yc + h/2) - float32(padY)) / scaleUsed

		x1 = clamp(x1, 0, float32(originalWidth))
		y1 = clamp(y1, 0, float32(originalHeight))
		x2 = clamp(x2, 0, float32(originalWidth))
		y2 = clamp(y2, 0, float32(originalHeight))

		if x2 <= x1 || y2 <= y1 {
			continue
		}

		// 使用中文标签 [1,2](@ref)
		englishLabel := yoloClasses[classID]
		chineseLabel := getChineseLabel(englishLabel)

		boundingBoxes = append(boundingBoxes, boundingBox{
			label:      chineseLabel, // 使用中文标签
			confidence: maxProb,
			x1:         x1, y1: y1, x2: x2, y2: y2,
		})
	}

	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	return nonMaxSuppression(boundingBoxes, iouThresh)
}

func nonMaxSuppression(boxes []boundingBox, iouThreshold float32) []boundingBox {
	if len(boxes) == 0 {
		return boxes
	}

	selected := make([]boundingBox, 0)
	picked := make([]bool, len(boxes))

	for i := 0; i < len(boxes); i++ {
		if picked[i] {
			continue
		}
		selected = append(selected, boxes[i])
		picked[i] = true

		for j := i + 1; j < len(boxes); j++ {
			if picked[j] || boxes[i].label != boxes[j].label {
				continue
			}
			iou := boxes[i].iou(&boxes[j])
			if iou > iouThreshold {
				picked[j] = true
			}
		}
	}
	return selected
}

func drawBoundingBoxesWithLabels(img image.Image, boxes []boundingBox, outputPath string) error {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, image.Point{}, draw.Src)

	// 定义不同类别的颜色映射
	colors := map[string]color.RGBA{
		"人员":    {255, 0, 0, 255},     // 红色
		"汽车":    {0, 255, 0, 255},     // 绿色
		"巴士":    {0, 0, 255, 255},     // 蓝色
		"摩托车":  {255, 255, 0, 255},   // 黄色
		"卡车":    {255, 0, 255, 255},   // 紫色
		"自行车":  {0, 255, 255, 255},   // 青色
		"default": {128, 128, 128, 255}, // 灰色(默认)
	}

	for _, box := range boxes {
		boxColor, exists := colors[box.label]
		if !exists {
			boxColor = colors["default"]
		}
		drawRectangle(rgba, box, boxColor)
		drawLabel(rgba, box, boxColor)
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	switch outputPath[len(outputPath)-4:] {
	case ".png":
		return png.Encode(outFile, rgba)
	default:
		return jpeg.Encode(outFile, rgba, &jpeg.Options{Quality: 95})
	}
}

func drawRectangle(img *image.RGBA, box boundingBox, color color.RGBA) {
	rect := box.toRect()
	thickness := 2

	for y := rect.Min.Y; y < rect.Min.Y+thickness && y <= rect.Max.Y; y++ {
		for x := rect.Min.X; x <= rect.Max.X; x++ {
			if x >= 0 && x < img.Bounds().Dx() && y >= 0 && y < img.Bounds().Dy() {
				img.Set(x, y, color)
			}
		}
	}
	for y := rect.Max.Y - thickness; y <= rect.Max.Y && y > rect.Min.Y; y++ {
		for x := rect.Min.X; x <= rect.Max.X; x++ {
			if x >= 0 && x < img.Bounds().Dx() && y >= 0 && y < img.Bounds().Dy() {
				img.Set(x, y, color)
			}
		}
	}
	for x := rect.Min.X; x < rect.Min.X+thickness && x <= rect.Max.X; x++ {
		for y := rect.Min.Y; y <= rect.Max.Y; y++ {
			if x >= 0 && x < img.Bounds().Dx() && y >= 0 && y < img.Bounds().Dy() {
				img.Set(x, y, color)
			}
		}
	}
	for x := rect.Max.X - thickness; x <= rect.Max.X && x > rect.Min.X; x++ {
		for y := rect.Min.Y; y <= rect.Max.Y; y++ {
			if x >= 0 && x < img.Bounds().Dx() && y >= 0 && y < img.Bounds().Dy() {
				img.Set(x, y, color)
			}
		}
	}
}

// 测量文本宽度和高度的辅助函数
func measureText(text string, face font.Face) (width, height int) {
	if face == nil {
		// 回退到估算（每个字符约14像素宽度，高度20像素）
		return len(text) * 14, 20
	}

	// 测量文本宽度
	drawer := &font.Drawer{Face: face}
	advance := drawer.MeasureString(text)
	width = advance.Round()

	// 获取字体高度
	metrics := face.Metrics()
	height = (metrics.Height + metrics.Descent).Round()

	return width, height
}

// 修改后的drawLabel函数，支持中文标签 [1,2](@ref)
func drawLabel(img *image.RGBA, box boundingBox, boxColor color.RGBA) {
	chineseLabel := getChineseLabel(box.label)
	labelText := fmt.Sprintf("%s %.2f", chineseLabel, box.confidence)
	rect := box.toRect()

	// 测量文本实际尺寸
	textWidth, textHeight := measureText(labelText, chineseFont)

	// 计算文本起始位置
	textX := rect.Min.X + 5
	textY := rect.Min.Y - 5

	// 上边界检查：确保文本不会超出图像顶部
	if textY < textHeight {
		textY = rect.Min.Y + textHeight + 5
	}

	// 下边界检查：确保文本不会超出图像底部
	imgHeight := img.Bounds().Dy()
	if textY > imgHeight-5 {
		textY = rect.Min.Y - textHeight - 5
		if textY < textHeight { // 如果上方空间也不足，放在框内
			textY = rect.Min.Y + 10
		}
	}

	// 右边界检查：确保文本不会超出图像右边界
	imgWidth := img.Bounds().Dx()
	if textX+textWidth > imgWidth-5 {
		textX = imgWidth - textWidth - 10 // 留出10像素边距
		if textX < 5 {                    // 如果文本太宽，至少保持5像素左边距
			textX = 5
			// 如果文本仍然太宽，可以截断或缩小字体，这里简单截断
			if textWidth > imgWidth-10 {
				maxChars := (imgWidth - 20) / 14 // 估算最大字符数
				if maxChars > 3 {
					labelText = labelText[:maxChars] + "..."
					textWidth, textHeight = measureText(labelText, chineseFont)
				}
			}
		}
	}

	// 左边界检查：确保文本不会超出左边界
	if textX < 5 {
		textX = 5
	}

	// 调整背景色块大小：根据实际文本尺寸，增加边距
	bgPadding := 8 // 背景边距
	bgWidth := textWidth + bgPadding*2
	bgHeight := textHeight + 4

	// 背景色块位置：文本位置减去边距
	bgX := textX - bgPadding/2
	bgY := textY - textHeight + 2 // 微调垂直位置使其居中

	// 确保背景色块不超出图像边界
	if bgX < 0 {
		bgX = 0
	}
	if bgX+bgWidth > imgWidth {
		bgX = imgWidth - bgWidth
		if bgX < 0 {
			bgX = 0
			bgWidth = imgWidth // 如果仍然太宽，使用最大宽度
		}
	}
	if bgY < 0 {
		bgY = 0
	}
	if bgY+bgHeight > imgHeight {
		bgY = imgHeight - bgHeight
	}

	// 绘制背景色块
	drawTextBackground(img, bgX, bgY, bgWidth, bgHeight, boxColor)

	// 绘制文本
	drawText(img, textX, textY, labelText, color.RGBA{255, 255, 255, 255})

	// 调试信息（可选）
	fmt.Printf("文本: %s, 尺寸: %dx%d, 位置: (%d,%d), 背景: (%d,%d)-%dx%d\n",
		labelText, textWidth, textHeight, textX, textY, bgX, bgY, bgWidth, bgHeight)
}

// 获取当前字体高度（基于字体大小）
func getFontHeight() int {
	// 字体大小为20时，估算实际渲染高度
	// 可以根据实际效果调整这个计算公式
	baseSize := 20                      // 当前字体大小
	return int(float64(baseSize) * 1.2) // 增加20%的行高补偿
}

// 精确计算文本位置
func getTextPosition(img *image.RGBA, box boundingBox, chineseFont font.Face) (int, int) {
	rect := box.toRect()

	// 获取字体度量
	metrics := chineseFont.Metrics()
	height := metrics.Height.Round() // 字体高度
	ascent := metrics.Ascent.Round() // 上坡度（基线以上高度）

	textX := rect.Min.X + 5

	// 计算文本Y坐标：框的上边界 - 上坡高度
	textY := rect.Min.Y - ascent

	// 边界检查
	if textY < height {
		textY = rect.Min.Y + height
	}
	if textY > img.Bounds().Dy()-5 {
		textY = rect.Min.Y - height/2
	}

	return textX, textY
}

// 改进的drawTextBackground函数，确保背景色块完全覆盖文本
func drawTextBackground(img *image.RGBA, x, y, width, height int, bgColor color.RGBA) {
	// 确保坐标在有效范围内
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x+width > img.Bounds().Dx() {
		width = img.Bounds().Dx() - x
	}
	if y+height > img.Bounds().Dy() {
		height = img.Bounds().Dy() - y
	}

	// 绘制实心矩形作为背景
	for i := x; i < x+width && i < img.Bounds().Dx(); i++ {
		for j := y; j < y+height && j < img.Bounds().Dy(); j++ {
			img.Set(i, j, bgColor)
		}
	}

	// 可选：添加边框使背景更明显
	borderColor := color.RGBA{
		uint8(float32(bgColor.R) * 0.7),
		uint8(float32(bgColor.G) * 0.7),
		uint8(float32(bgColor.B) * 0.7),
		255,
	}

	// 绘制上边框
	for i := x; i < x+width && i < img.Bounds().Dx(); i++ {
		for j := y; j < y+2 && j < img.Bounds().Dy(); j++ {
			img.Set(i, j, borderColor)
		}
	}
	// 绘制下边框
	for i := x; i < x+width && i < img.Bounds().Dx(); i++ {
		for j := y + height - 2; j < y+height && j < img.Bounds().Dy(); j++ {
			if j >= 0 {
				img.Set(i, j, borderColor)
			}
		}
	}
	// 绘制左边框
	for j := y; j < y+height && j < img.Bounds().Dy(); j++ {
		for i := x; i < x+2 && i < img.Bounds().Dx(); i++ {
			img.Set(i, j, borderColor)
		}
	}
	// 绘制右边框
	for j := y; j < y+height && j < img.Bounds().Dy(); j++ {
		for i := x + width - 2; i < x+width && i < img.Bounds().Dx(); i++ {
			if i >= 0 {
				img.Set(i, j, borderColor)
			}
		}
	}
}

// 辅助函数：测量文本宽度（近似）
func measureTextWidth(text string) int {
	// 中文字符通常比英文字符宽
	width := 0
	for _, r := range text {
		if r > 127 { // 中文字符
			width += 14
		} else { // 英文字符
			width += 7
		}
	}
	return width
}

// 修改后的drawText函数，支持中文显示
func drawText(img *image.RGBA, x, y int, text string, textColor color.RGBA) {
	point := fixed.P(x, y)

	d := &font.Drawer{
		Dst: img,
		Src: image.NewUniform(textColor),
		Dot: point,
	}

	// 优先使用加载的中文字体
	if chineseFont != nil {
		d.Face = chineseFont
	} else {
		// 关键修正：回退到真正的默认字体 face，而不是一个函数
		// 使用 inconsolata.Regular8x16，它是一个实现了 font.Face 接口的类型
		d.Face = inconsolata.Regular8x16
		log.Println("警告: 使用默认等宽字体 inconsolata，中文可能显示为方框或乱码")
	}

	d.DrawString(text)
}

// 更精确的字体度量函数
func getFontMetrics(face font.Face) (ascent, descent, height int) {
	if face == nil {
		return 16, 4, 20 // 默认值
	}

	metrics := face.Metrics()
	ascent = metrics.Ascent.Round()
	descent = metrics.Descent.Round()
	height = metrics.Height.Round()

	return ascent, descent, height
}

func clamp(value, min, max float32) float32 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

// YOLO类别标签（英文原始标签）
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// 中英文标签映射表
var objectMap = map[string]string{
	"person":         "人员",
	"bicycle":        "自行车",
	"car":            "汽车",
	"motorcycle":     "摩托车",
	"airplane":       "飞机",
	"bus":            "巴士",
	"train":          "火车",
	"truck":          "卡车",
	"boat":           "船",
	"traffic light":  "交通灯",
	"fire hydrant":   "消防栓",
	"stop sign":      "停止标志",
	"parking meter":  "停车计时器",
	"bench":          "长凳",
	"bird":           "鸟",
	"cat":            "猫",
	"dog":            "狗",
	"horse":          "马",
	"sheep":          "羊",
	"cow":            "牛",
	"elephant":       "大象",
	"bear":           "熊",
	"zebra":          "斑马",
	"giraffe":        "长颈鹿",
	"backpack":       "背包",
	"umbrella":       "雨伞",
	"handbag":        "手提包",
	"tie":            "领带",
	"suitcase":       "行李箱",
	"frisbee":        "飞盘",
	"skis":           "滑雪板",
	"snowboard":      "滑雪板",
	"sports ball":    "运动球",
	"kite":           "风筝",
	"baseball bat":   "棒球棒",
	"baseball glove": "棒球手套",
	"skateboard":     "滑板",
	"surfboard":      "冲浪板",
	"tennis racket":  "网球拍",
	"bottle":         "瓶子",
	"wine glass":     "酒杯",
	"cup":            "杯子",
	"fork":           "叉子",
	"knife":          "刀",
	"spoon":          "勺子",
	"bowl":           "碗",
	"banana":         "香蕉",
	"apple":          "苹果",
	"sandwich":       "三明治",
	"orange":         "橙子",
	"broccoli":       "西兰花",
	"carrot":         "胡萝卜",
	"hot dog":        "热狗",
	"pizza":          "披萨",
	"donut":          "甜甜圈",
	"cake":           "蛋糕",
	"chair":          "椅子",
	"couch":          "沙发",
	"potted plant":   "盆栽植物",
	"bed":            "床",
	"dining table":   "餐桌",
	"toilet":         "马桶",
	"tv":             "电视",
	"laptop":         "笔记本电脑",
	"mouse":          "鼠标",
	"remote":         "遥控器",
	"keyboard":       "键盘",
	"cell phone":     "手机",
	"microwave":      "微波炉",
	"oven":           "烤箱",
	"toaster":        "烤面包机",
	"sink":           "水槽",
	"refrigerator":   "冰箱",
	"book":           "书",
	"clock":          "时钟",
	"vase":           "花瓶",
	"scissors":       "剪刀",
	"teddy bear":     "泰迪熊",
	"hair drier":     "吹风机",
	"toothbrush":     "牙刷",
}
