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
	modelPath       = "D:\\\\mlz\\\\go\\\\src\\\\yolo\\\\yolo11x.onnx"
	imagePath       = "D:\\\\mlz\\\\go\\\\src\\\\yolo\\bus.jpg"
	outputImagePath = "D:\\\\mlz\\\\go\\\\src\\\\yolo\\bus-yolo11x.onnx.jpg"
	useCoreML       = false

	//高召回要求（如安防） 降低 conf（0.2~0.3），提高 iou（0.6~0.7）
	confidenceThreshold = flag.Float64("conf", 0.25, "置信度阈值")
	iouThreshold        = flag.Float64("iou", 0.7, "IOU阈值")
	modelInputSize      = flag.Int("size", 640, "模型输入尺寸")
	// rect	bool	True	如果启用，则对图像较短的一边进行最小填充，直到可以被步长整除，以提高推理速度。如果禁用，则在推理期间将图像填充为正方形。
	useRectScaling = flag.Bool("rect", false, "是否使用矩形缩放（保持长宽比）")

	// 系统文本位置参数
	systemTextLocation = flag.String("text-location", "bottom-left", "系统文本位置 (top-left, bottom-left, top-right, bottom-right)")
	systemTextContent  = flag.String("system-text", "重要设施危险场景监测系统", "系统显示文本")
	systemTextEnabled  = flag.Bool("enable-system-text", true, "是否显示系统文本")

	// 中文字体变量
	chineseFont font.Face

	// ONNX Runtime 初始化状态控制
	ortInitialized bool
	ortInitMutex   sync.Mutex
)

// 缩放和填充信息结构体，用于坐标转换
type ScaleInfo struct {
	ScaleX    float32
	ScaleY    float32
	PadLeft   int
	PadTop    int
	NewWidth  int
	NewHeight int
}

func main() {
	// 设置环境变量确保UTF-8编码支持
	os.Setenv("LC_ALL", "zh_CN.UTF-8")

	flag.Parse()
	fmt.Printf("使用参数: conf=%.2f, iou=%.2f, size=%d, useRectScaling=%v\n", *confidenceThreshold, *iouThreshold, *modelInputSize, *useRectScaling)
	os.Exit(run())

}

func run() int {
	// 初始化中文字体
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

	// 修改：prepareInput现在返回缩放信息
	scaleInfo, e := prepareInput(originalPic, modelSession.Input)
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

	// 修改：传递scaleInfo到processOutput
	allBoxes := processOutput(modelSession.Output.GetData(), originalWidth, originalHeight,
		float32(*confidenceThreshold), float32(*iouThreshold), scaleInfo)

	fmt.Printf("检测到 %d 个对象:\n", len(allBoxes))
	for i, box := range allBoxes {
		chineseLabel := getChineseLabel(box.label)
		fmt.Printf("框 %d: %s (置信度: %.6f): (%.1f, %.1f, %.1f, %.1f)\n",
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

// 计算颜色亮度的函数
func getLuminance(c color.RGBA) float64 {
	return 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
}

// 获取高对比度文本颜色
func getContrastTextColor(backgroundColor color.RGBA) color.RGBA {
	luminance := getLuminance(backgroundColor)
	if luminance > 128 {
		return color.RGBA{0, 0, 0, 255} // 深色文本（黑色）
	}
	return color.RGBA{255, 255, 255, 255} // 浅色文本（白色）
}

// 获取区域平均颜色（用于系统文本背景）
func getAreaAverageColor(img *image.RGBA, rect image.Rectangle) color.RGBA {
	var r, g, b, count uint32
	count = 0

	for y := rect.Min.Y; y < rect.Max.Y && y < img.Bounds().Dy(); y++ {
		for x := rect.Min.X; x < rect.Max.X && x < img.Bounds().Dx(); x++ {
			c := color.RGBAModel.Convert(img.At(x, y)).(color.RGBA)
			r += uint32(c.R)
			g += uint32(c.G)
			b += uint32(c.B)
			count++
		}
	}

	if count == 0 {
		return color.RGBA{0, 0, 0, 180} // 默认半透明黑色背景
	}

	return color.RGBA{
		uint8(r / count),
		uint8(g / count),
		uint8(b / count),
		180, // 半透明
	}
}

// 绘制系统文本函数
func drawSystemText(img *image.RGBA, location string) {
	if !*systemTextEnabled || *systemTextContent == "" {
		return
	}

	text := *systemTextContent
	bounds := img.Bounds()
	textWidth, textHeight := measureText(text, chineseFont)

	// 设置边距
	margin := 15
	bgPadding := 10

	// 计算文本位置
	var textX, textY int
	var bgRect image.Rectangle

	switch location {
	case "top-left":
		textX = margin
		textY = margin + textHeight
		bgRect = image.Rect(
			textX-bgPadding,
			textY-textHeight-bgPadding/2,
			textX+textWidth+bgPadding,
			textY+bgPadding/2,
		)
	case "top-right":
		textX = bounds.Dx() - textWidth - margin
		textY = margin + textHeight
		bgRect = image.Rect(
			textX-bgPadding,
			textY-textHeight-bgPadding/2,
			textX+textWidth+bgPadding,
			textY+bgPadding/2,
		)
	case "bottom-right":
		textX = bounds.Dx() - textWidth - margin
		textY = bounds.Dy() - margin
		bgRect = image.Rect(
			textX-bgPadding,
			textY-textHeight-bgPadding/2,
			textX+textWidth+bgPadding,
			textY+bgPadding/2,
		)
	default: // bottom-left (默认)
		textX = margin
		textY = bounds.Dy() - margin
		bgRect = image.Rect(
			textX-bgPadding,
			textY-textHeight-bgPadding/2,
			textX+textWidth+bgPadding,
			textY+bgPadding/2,
		)
	}

	// 确保背景矩形在图像范围内
	if bgRect.Min.X < 0 {
		bgRect.Min.X = 0
	}
	if bgRect.Min.Y < 0 {
		bgRect.Min.Y = 0
	}
	if bgRect.Max.X > bounds.Dx() {
		bgRect.Max.X = bounds.Dx()
	}
	if bgRect.Max.Y > bounds.Dy() {
		bgRect.Max.Y = bounds.Dy()
	}

	// 获取背景区域平均颜色
	bgColor := getAreaAverageColor(img, bgRect)

	// 根据背景亮度选择文本颜色
	textColor := getContrastTextColor(bgColor)

	// 绘制半透明背景
	drawTextBackground(img, bgRect.Min.X, bgRect.Min.Y,
		bgRect.Dx(), bgRect.Dy(), bgColor)

	// 绘制系统文本
	drawText(img, textX, textY, text, textColor)
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
		Size:    18,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return fmt.Errorf("创建字体face失败: %w", err)
	}

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
	if chinese, exists := detectLabeltMap[englishLabel]; exists {
		return chinese
	}
	return englishLabel
}

// 安全的ONNX Runtime环境初始化函数
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

// 修改：修复矩形缩放函数，确保符合官方要求
func resizeWithAspectRatio(img image.Image, targetWidth, targetHeight int) image.Image {
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	// 计算缩放比例，保持长宽比
	scale := float64(targetWidth) / float64(originalWidth)
	if float64(targetHeight)/float64(originalHeight) < scale {
		scale = float64(targetHeight) / float64(originalHeight)
	}

	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)
	result := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))

	// 填充灰色背景 (114, 114, 114) - YOLO标准
	draw.Draw(result, result.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)

	// 将缩放后的图像居中放置
	offsetX := (targetWidth - newWidth) / 2
	offsetY := (targetHeight - newHeight) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+newWidth, offsetY+newHeight),
		resized, image.Point{}, draw.Src)

	return result
}

// 矩形缩放模式（rect=true）的专用函数
func resizeWithRectScaling(img image.Image, targetSize int) (image.Image, ScaleInfo) {
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	// 计算缩放比例，以长边为基准
	scale := float64(targetSize) / float64(max(originalWidth, originalHeight))
	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	// 关键修复：确保尺寸能被32整除（YOLO的步长要求）
	// 这是官方rect=true模式的核心要求[6](@ref)
	stride := 32
	newWidth = (newWidth + stride/2) / stride * stride
	newHeight = (newHeight + stride/2) / stride * stride

	// 确保不超过目标尺寸
	newWidth = min(newWidth, targetSize)
	newHeight = min(newHeight, targetSize)

	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)
	result := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))

	// 填充灰色背景
	draw.Draw(result, result.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)

	// 将缩放后的图像放置在左上角（矩形缩放模式）
	draw.Draw(result, image.Rect(0, 0, newWidth, newHeight), resized, image.Point{}, draw.Src)

	scaleInfo := ScaleInfo{
		ScaleX:    float32(scale),
		ScaleY:    float32(scale),
		PadLeft:   0, // 矩形模式填充在左上角
		PadTop:    0,
		NewWidth:  newWidth,
		NewHeight: newHeight,
	}

	return result, scaleInfo
}

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("无法找到支持此系统的onnxruntime库版本")
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

// 修改：processOutput现在接受scaleInfo参数
func processOutput(output []float32, originalWidth, originalHeight int, confThreshold, iouThresh float32, scaleInfo ScaleInfo) []boundingBox {
	boundingBoxes := make([]boundingBox, 0, 8400)

	// 使用传入的scaleInfo而不是重新计算
	scaleX := scaleInfo.ScaleX
	scaleY := scaleInfo.ScaleY
	padX := scaleInfo.PadLeft
	padY := scaleInfo.PadTop

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

		// 统一的坐标转换公式，使用正确的缩放和填充参数
		x1 := (xc - w/2 - float32(padX)) / scaleX
		y1 := (yc - h/2 - float32(padY)) / scaleY
		x2 := (xc + w/2 - float32(padX)) / scaleX
		y2 := (yc + h/2 - float32(padY)) / scaleY

		// 边界约束
		x1 = clamp(x1, 0, float32(originalWidth))
		y1 = clamp(y1, 0, float32(originalHeight))
		x2 = clamp(x2, 0, float32(originalWidth))
		y2 = clamp(y2, 0, float32(originalHeight))

		if x2 <= x1 || y2 <= y1 {
			continue
		}

		englishLabel := yoloClasses[classID]

		boundingBoxes = append(boundingBoxes, boundingBox{
			label:      englishLabel,
			confidence: maxProb,
			x1:         x1, y1: y1, x2: x2, y2: y2,
		})
	}

	// 按置信度排序
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	return nonMaxSuppression(boundingBoxes, iouThresh)
}

// 修改：prepareInput现在返回ScaleInfo
func prepareInput(pic image.Image, dst *ort.Tensor[float32]) (ScaleInfo, error) {
	data := dst.GetData()
	inputSize := *modelInputSize
	channelSize := inputSize * inputSize

	if len(data) < (channelSize * 3) {
		return ScaleInfo{}, fmt.Errorf("目标张量仅包含 %d 个浮点数，需要 %d", len(data), channelSize*3)
	}

	redChannel := data[0:channelSize]
	greenChannel := data[channelSize : channelSize*2]
	blueChannel := data[channelSize*2 : channelSize*3]

	var resizedImg image.Image
	var scaleInfo ScaleInfo

	if *useRectScaling {
		// rect=true: 使用矩形缩放模式（官方正确实现）
		var err error
		resizedImg, scaleInfo = resizeWithRectScaling(pic, inputSize)
		if err != nil {
			return ScaleInfo{}, err
		}
		fmt.Printf("矩形缩放模式: 缩放后尺寸 %dx%d, 缩放比例 (%.4f, %.4f), 填充 (%d, %d)\n",
			scaleInfo.NewWidth, scaleInfo.NewHeight, scaleInfo.ScaleX, scaleInfo.ScaleY, scaleInfo.PadLeft, scaleInfo.PadTop)
	} else {
		// rect=false: 使用保持长宽比的居中填充（修正拉伸错误）
		resizedImg = resizeWithAspectRatio(pic, inputSize, inputSize)

		// 计算scaleInfo用于坐标转换
		bounds := pic.Bounds()
		originalWidth := bounds.Dx()
		originalHeight := bounds.Dy()

		scale := float32(inputSize) / float32(max(originalWidth, originalHeight))
		newWidth := int(float32(originalWidth) * scale)
		newHeight := int(float32(originalHeight) * scale)

		scaleInfo = ScaleInfo{
			ScaleX:    scale,
			ScaleY:    scale,
			PadLeft:   (inputSize - newWidth) / 2,
			PadTop:    (inputSize - newHeight) / 2,
			NewWidth:  newWidth,
			NewHeight: newHeight,
		}
		fmt.Printf("居中填充模式: 缩放后尺寸 %dx%d, 缩放比例 (%.4f, %.4f), 填充 (%d, %d)\n",
			scaleInfo.NewWidth, scaleInfo.NewHeight, scaleInfo.ScaleX, scaleInfo.ScaleY, scaleInfo.PadLeft, scaleInfo.PadTop)
	}

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
	return scaleInfo, nil
}

// 确保 clamp 函数存在
func clamp(value, min, max float32) float32 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
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

	// 定义不同类别的颜色映射 "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "horse", "sheep", "cow"
	var colors = map[string]color.RGBA{
		"person":         {255, 0, 0, 255},     // 红色
		"car":            {0, 255, 0, 255},     // 绿色
		"bus":            {0, 0, 255, 255},     // 蓝色
		"motorcycle":     {255, 255, 0, 255},   // 黄色
		"truck":          {255, 0, 255, 255},   // 紫色
		"bicycle":        {0, 255, 255, 255},   // 青色
		"airplane":       {255, 0, 255, 255},   // 紫色
		"train":          {128, 0, 0, 255},     // 深红色
		"boat":           {0, 0, 128, 255},     // 深蓝色
		"traffic light":  {128, 128, 0, 255},   // 橄榄绿
		"fire hydrant":   {128, 0, 128, 255},   // 深紫色
		"stop sign":      {0, 128, 128, 255},   // 深青色
		"parking meter":  {0, 255, 255, 255},   // 橙色
		"bench":          {255, 0, 128, 255},   // 粉色
		"bird":           {0, 255, 128, 255},   // 浅青色
		"cat":            {128, 255, 0, 255},   // 浅黄绿色
		"dog":            {128, 0, 255, 255},   // 紫罗兰色
		"horse":          {0, 128, 255, 255},   // 天蓝色
		"sheep":          {255, 128, 128, 255}, // 浅红色
		"cow":            {128, 255, 128, 255}, // 浅绿色
		"elephant":       {128, 128, 255, 255}, // 浅蓝色
		"bear":           {255, 255, 128, 255}, // 浅黄色
		"zebra":          {255, 128, 255, 255}, // 浅紫色
		"giraffe":        {128, 255, 255, 255}, // 浅青色
		"backpack":       {64, 0, 0, 255},      // 暗红色
		"umbrella":       {0, 64, 0, 255},      // 暗绿色
		"handbag":        {0, 0, 64, 255},      // 暗蓝色
		"tie":            {64, 64, 0, 255},     // 暗橄榄绿
		"suitcase":       {64, 0, 64, 255},     // 暗紫色
		"frisbee":        {0, 64, 64, 255},     // 暗青色
		"skis":           {255, 64, 0, 255},    // 深橙色
		"snowboard":      {255, 0, 64, 255},    // 深粉色
		"sports ball":    {0, 255, 64, 255},    // 深浅青色
		"kite":           {64, 255, 0, 255},    // 深黄绿色
		"baseball bat":   {64, 0, 255, 255},    // 深紫罗兰色
		"baseball glove": {0, 64, 255, 255},    // 深天蓝色
		"skateboard":     {255, 64, 64, 255},   // 深浅红色
		"surfboard":      {64, 255, 64, 255},   // 深浅绿色
		"tennis racket":  {64, 64, 255, 255},   // 深浅蓝色
		"bottle":         {255, 255, 64, 255},  // 深浅黄色
		"wine glass":     {255, 64, 255, 255},  // 深浅紫色
		"cup":            {64, 255, 255, 255},  // 深浅青色
		"fork":           {192, 0, 0, 255},     // 枣红色
		"knife":          {0, 192, 0, 255},     // 鲜绿色
		"spoon":          {0, 0, 192, 255},     // 宝蓝色
		"bowl":           {192, 192, 0, 255},   // 金黄色
		"banana":         {192, 0, 192, 255},   // 紫红色
		"apple":          {0, 192, 192, 255},   // 碧绿色
		"sandwich":       {255, 192, 0, 255},   // 橙黄色
		"orange":         {255, 0, 192, 255},   // 玫红色
		"broccoli":       {0, 255, 192, 255},   // 薄荷绿
		"carrot":         {192, 255, 0, 255},   // 黄绿色
		"hot dog":        {192, 0, 255, 255},   // 靛蓝色
		"pizza":          {0, 192, 255, 255},   // 淡蓝色
		"donut":          {255, 192, 192, 255}, // 粉红色
		"cake":           {192, 255, 192, 255}, // 嫩绿色
		"chair":          {192, 192, 255, 255}, // 淡紫色
		"couch":          {255, 255, 192, 255}, // 米黄色
		"potted plant":   {255, 192, 255, 255}, // 淡粉色
		"bed":            {192, 255, 255, 255}, // 淡青色
		"dining table":   {160, 0, 0, 255},     // 赤红色
		"toilet":         {0, 160, 0, 255},     // 翠绿色
		"tv":             {0, 0, 160, 255},     // 藏蓝色
		"laptop":         {160, 160, 0, 255},   // 土黄色
		"mouse":          {160, 0, 160, 255},   // 深粉色
		"remote":         {0, 160, 160, 255},   // 青绿色
		"keyboard":       {255, 160, 0, 255},   // 橙红色
		"cell phone":     {255, 0, 160, 255},   // 桃红色
		"microwave":      {0, 255, 160, 255},   // 水绿色
		"oven":           {160, 255, 0, 255},   // 草绿色
		"toaster":        {160, 0, 255, 255},   // 蓝紫色
		"sink":           {0, 160, 255, 255},   // 海蓝色
		"refrigerator":   {255, 160, 160, 255}, // 珊瑚色
		"book":           {160, 255, 160, 255}, // 豆绿色
		"clock":          {160, 160, 255, 255}, // 蓝灰色
		"vase":           {255, 255, 160, 255}, // 鹅黄色
		"scissors":       {255, 160, 255, 255}, // 粉紫色
		"teddy bear":     {160, 255, 255, 255}, // 天青色
		"hair drier":     {96, 96, 96, 255},    // 深灰色
		"toothbrush":     {200, 200, 200, 255}, // 浅灰色
		"default":        {128, 128, 128, 255}, // 灰色(默认)
	}

	for _, box := range boxes {
		boxColor, exists := colors[box.label]
		if !exists {
			boxColor = colors["default"]
		}
		drawRectangle(rgba, box, boxColor)
		drawLabel(rgba, box, boxColor)
	}

	// 绘制系统文本
	drawSystemText(rgba, *systemTextLocation)

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
		return len(text) * 14, 20
	}

	drawer := &font.Drawer{Face: face}
	advance := drawer.MeasureString(text)
	width = advance.Round()

	metrics := face.Metrics()
	height = (metrics.Height + metrics.Descent).Round()

	return width, height
}

// 修改后的drawLabel函数，支持中文标签
func drawLabel(img *image.RGBA, box boundingBox, boxColor color.RGBA) {
	chineseLabel := getChineseLabel(box.label)
	labelText := fmt.Sprintf("%s %.2f", chineseLabel, box.confidence)
	rect := box.toRect()

	textWidth, textHeight := measureText(labelText, chineseFont)

	textX := rect.Min.X + 5
	textY := rect.Min.Y - 5

	imgHeight := img.Bounds().Dy()
	if textY < textHeight {
		textY = rect.Min.Y + textHeight + 5
	}
	if textY > imgHeight-5 {
		textY = rect.Min.Y - textHeight - 5
		if textY < textHeight {
			textY = rect.Min.Y + 10
		}
	}

	imgWidth := img.Bounds().Dx()
	if textX+textWidth > imgWidth-5 {
		textX = imgWidth - textWidth - 10
		if textX < 5 {
			textX = 5
			if textWidth > imgWidth-10 {
				maxChars := (imgWidth - 20) / 14
				if maxChars > 3 {
					labelText = labelText[:maxChars] + "..."
					textWidth, textHeight = measureText(labelText, chineseFont)
				}
			}
		}
	}
	if textX < 5 {
		textX = 5
	}

	bgPadding := 8
	bgWidth := textWidth + bgPadding*2
	bgHeight := textHeight + 4

	bgX := textX - bgPadding/2
	bgY := textY - textHeight + 2

	if bgX < 0 {
		bgX = 0
	}
	if bgX+bgWidth > imgWidth {
		bgX = imgWidth - bgWidth
		if bgX < 0 {
			bgX = 0
			bgWidth = imgWidth
		}
	}
	if bgY < 0 {
		bgY = 0
	}
	if bgY+bgHeight > imgHeight {
		bgY = imgHeight - bgHeight
	}

	textColor := getContrastTextColor(boxColor)

	drawTextBackground(img, bgX, bgY, bgWidth, bgHeight, boxColor)
	drawText(img, textX, textY, labelText, textColor)
}

// 改进的drawTextBackground函数
func drawTextBackground(img *image.RGBA, x, y, width, height int, bgColor color.RGBA) {
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

	for i := x; i < x+width && i < img.Bounds().Dx(); i++ {
		for j := y; j < y+height && j < img.Bounds().Dy(); j++ {
			img.Set(i, j, bgColor)
		}
	}
}

// 修改后的drawText函数，支持中文显示
func drawText(img *image.RGBA, x, y int, text string, textColor color.RGBA) {
	point := fixed.P(x, y)

	d := &font.Drawer{
		Dst: img,
		Src: image.NewUniform(textColor),
		Dot: point,
	}

	if chineseFont != nil {
		d.Face = chineseFont
	} else {
		d.Face = inconsolata.Regular8x16
	}

	d.DrawString(text)
}

// ensureDirForFile 确保给定文件路径的父目录存在
func ensureDirForFile(filePath string) error {
	dir := filepath.Dir(filePath)

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err = os.MkdirAll(dir, 0755)
		if err != nil {
			return fmt.Errorf("无法创建目录 %s: %w", dir, err)
		}
	} else if err != nil {
		return fmt.Errorf("检查目录状态失败 %s: %w", dir, err)
	}
	return nil
}

func checkStrIsInArray(target string, str_array []string) bool {
	sort.Strings(str_array)
	index := sort.SearchStrings(str_array, target)
	if index < len(str_array) && str_array[index] == target {
		return true
	}
	return false
}

// YOLO类别标签（英文原始标签）[1,2](@ref)
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
var detectLabeltMap = map[string]string{
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
