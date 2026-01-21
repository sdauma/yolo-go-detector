// Package main 实现基于ONNXRuntime的YOLO目标检测程序
// 该程序支持多种输入格式（图像、目录、视频）、批量处理、中文标签显示等功能
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/flopp/go-findfont" // 添加字体查找库
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/font"
	"golang.org/x/image/font/inconsolata" // 用于回退的默认字体
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// 全局配置参数
var (
<<<<<<< Updated upstream
	// 模型路径配置
	modelPath = "./third_party/yolo11x.onnx" // YOLO模型文件路径
	useCoreML = false                        // 是否使用CoreML加速（仅限iOS/macOS）

	// 输入输出路径参数
	inputImagePath  = flag.String("img", "./assets/bus.jpg", "输入图像路径、目录、视频文件或.txt文件")
	outputImagePath = flag.String("output", "./assets/bus_11x_false.jpg", "输出图像路径（仅在输入单个图像时有效）")

	// 检测参数配置
	confidenceThreshold = flag.Float64("conf", 0.25, "置信度阈值，过滤低置信度检测结果")
	iouThreshold        = flag.Float64("iou", 0.7, "IOU阈值，用于非极大值抑制(NMS)")
	modelInputSize      = flag.Int("size", 640, "模型输入尺寸，通常为640x640")
	// rect	bool	True	如果启用，则对图像较短的一边进行最小填充，直到可以被步长整除，以提高推理速度。如果禁用，则在推理期间将图像填充为正方形。
	useRectScaling = flag.Bool("rect", false, "是否使用矩形缩放（保持长宽比）")
	// augment	bool	False	启用测试时增强 (TTA) 进行预测，可能会提高检测的鲁棒性，但会降低推理速度。
	useAugment = flag.Bool("augment", false, "是否启用测试时增强 (TTA) 进行预测")
	// batch	int	1	指定推理的批处理大小（仅在源为以下情况时有效： 一个目录、视频文件，或 .txt 文件)。
	batchSize = flag.Int("batch", 1, "指定推理的批处理大小")

	// 系统显示参数（用于监控系统等应用场景）
	systemTextLocation = flag.String("text-location", "bottom-left", "系统文本位置 (top-left, bottom-left, top-right, bottom-right)")
	systemTextContent  = flag.String("system-text", "重要设施危险场景监测系统", "系统显示文本")
	systemTextEnabled  = flag.Bool("enable-system-text", true, "是否显示系统文本")

	// 并发处理相关参数
	workerCount = flag.Int("workers", max(1, runtime.NumCPU()/2), "并发工作协程数量")
	queueSize   = flag.Int("queue-size", 100, "任务队列大小")
	taskTimeout = flag.Duration("timeout", 30*time.Second, "单个任务超时时间")
=======
	modelPath       = "D:\\mlz\\go\\src\\yolo\\yoloe-v8l-seg-pf.onnx"
	imagePath       = "D:\\mlz\\go\\src\\qwen\\1.jpg"
	outputImagePath = "D:\\mlz\\go\\src\\qwen\\2.jpg"
	useCoreML       = false

	//高召回要求（如安防） 降低 conf（0.2~0.3），提高 iou（0.6~0.7）
	confidenceThreshold = flag.Float64("conf", 0.25, "置信度阈值")
	iouThreshold        = flag.Float64("iou", 0.7, "IOU阈值")
	modelInputSize      = flag.Int("size", 640, "模型输入尺寸")
	useRectScaling      = flag.Bool("rect", true, "是否使用矩形缩放（保持长宽比）") // 新增rect参数
	// 新增：系统文本位置参数
	systemTextLocation = flag.String("text-location", "bottom-left", "系统文本位置 (top-left, bottom-left, top-right, bottom-right)")
	systemTextContent  = flag.String("system-text", "野外环境重点设施危险场景检测系统", "系统显示文本")
	systemTextEnabled  = flag.Bool("enable-system-text", true, "是否显示系统文本")
>>>>>>> Stashed changes

	// 中文字体变量
	chineseFont font.Face

	// ONNX Runtime 初始化状态控制（线程安全）
	ortInitialized bool
	ortInitMutex   sync.Mutex
)

// 定义支持的图像和视频扩展名常量，提升可维护性
var (
	supportedImageExts = map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
		".gif":  true,
	}
	supportedVideoExts = map[string]bool{
		".mp4": true,
		".avi": true,
		".mov": true,
		".mkv": true,
	}
)

// 缩放和填充信息结构体，用于坐标转换
// 在图像预处理过程中记录缩放参数，以便将模型输出坐标转换回原图坐标
type ScaleInfo struct {
	ScaleX    float32 // X轴缩放比例
	ScaleY    float32 // Y轴缩放比例
	PadLeft   int     // 左侧填充像素数
	PadTop    int     // 顶部填充像素数
	NewWidth  int     // 缩放后宽度
	NewHeight int     // 缩放后高度
}

// 主函数：程序入口点
// 解析命令行参数，初始化配置，根据输入类型决定处理方式
func main() {
	// 设置环境变量确保UTF-8编码支持
	os.Setenv("LC_ALL", "zh_CN.UTF-8")

	flag.Parse()
<<<<<<< Updated upstream
	fmt.Printf("使用参数: conf=%.2f, iou=%.2f, size=%d, rect=%t, augment=%t, batch=%d, workers=%d\n",
		*confidenceThreshold, *iouThreshold, *modelInputSize, *useRectScaling, *useAugment, *batchSize, *workerCount)

	// 创建默认输出目录
	defaultOutputDir := "./assets"
	if _, err := os.Stat(defaultOutputDir); os.IsNotExist(err) {
		err = os.Mkdir(defaultOutputDir, 0755)
		if err != nil {
			fmt.Printf("创建输出目录失败: %v\n", err)
			return
		}
	}

	// 获取所有图像路径
	imagePaths, err := getImagePaths(*inputImagePath)
	if err != nil {
		fmt.Printf("获取图像路径失败: %v\n", err)
		return
	}

	if len(imagePaths) == 0 {
		fmt.Printf("未找到任何图像文件\n")
		return
	}

	// 检查输入是否是目录
	isInputDirectory := false
	if fileInfo, err := os.Stat(*inputImagePath); err == nil && fileInfo.IsDir() {
		isInputDirectory = true
	}

	if len(imagePaths) == 1 && !isInputDirectory {
		// 单个图像，使用指定的输出路径
		fmt.Printf("找到 1 个图像文件，使用指定的输出路径: %s\n", *outputImagePath)

		// 执行检测
		num, desc, err := detectImage(imagePaths[0], *outputImagePath)
		if err != nil {
			fmt.Printf("处理图像 %s 时出错: %v\n", imagePaths[0], err)
		} else {
			fmt.Printf("图像 %s 检测完成: %d 个对象 - %s\n", imagePaths[0], num, desc)
			fmt.Printf("检测结果已保存至: %s\n", *outputImagePath)
		}
	} else if isInputDirectory {
		// 输入是目录的情况，使用目录处理函数
		err := ProcessImageDirectory(*inputImagePath, defaultOutputDir)
		if err != nil {
			fmt.Printf("处理目录时出错: %v\n", err)
		} else {
			fmt.Printf("目录处理完成\n")
		}
	} else {
		// 多个图像（来自txt文件等），使用批量处理逻辑
		fmt.Printf("找到 %d 个图像文件，将使用并发处理（工作协程: %d）\n", len(imagePaths), *workerCount)

		// 生成输出路径列表
		outputPaths := make([]string, len(imagePaths))
		for i, imagePath := range imagePaths {
			imgName := filepath.Base(imagePath)
			outputPaths[i] = filepath.Join(defaultOutputDir, "detected_"+imgName)
		}

		// 使用并发处理图像
		err := ConcurrentBatchProcessImages(imagePaths, outputPaths)
		if err != nil {
			fmt.Printf("批量处理出错: %v\n", err)
		}
	}

	fmt.Printf("所有图像处理完成\n")
}

// 多协程批量处理图片的函数
func ConcurrentBatchProcessImages(sourceImagePaths []string, outputImagePaths []string) error {
	if len(sourceImagePaths) != len(outputImagePaths) {
		return fmt.Errorf("输入图片路径数量(%d)与输出图片路径数量(%d)不匹配", len(sourceImagePaths), len(outputImagePaths))
	}

	fmt.Printf("启动并发处理，工作协程数量: %d, 队列大小: %d\n", *workerCount, *queueSize)

	// 创建视频检测管理器
	manager := NewVideoDetectorManager(*workerCount, *queueSize, *taskTimeout)
	defer manager.Stop()

	// 创建任务列表
	imagePaths := make([]string, len(sourceImagePaths))
	copy(imagePaths, sourceImagePaths)

	// 提交所有任务
	results := manager.ProcessImageBatch(imagePaths)

	// 处理结果并保存检测结果
	for i, result := range results {
		if result.Error != nil {
			fmt.Printf("处理图像 %s 时出错: %v\n", result.ImagePath, result.Error)
		} else {
			outputPath := outputImagePaths[i]

			// 将检测结果绘制到图像
			originalPic, err := loadImageFile(result.ImagePath)
			if err != nil {
				fmt.Printf("加载原图失败 %s: %v\n", result.ImagePath, err)
				continue
			}

			err = drawBoundingBoxesWithLabels(originalPic, result.Objects, outputPath)
			if err != nil {
				fmt.Printf("绘制边界框失败 %s: %v\n", result.ImagePath, err)
				continue
			}

			fmt.Printf("图像 %s 检测完成: %d 个对象，已保存至 %s\n", result.ImagePath, len(result.Objects), outputPath)
		}
	}

	return nil
}

// 获取输入源的所有图像路径
// 支持多种输入类型：单个图像、目录（一级）、文本文件列表
// inputSource: 输入源路径（文件/目录/.txt文件）
// return: 图像路径列表 + 错误信息
func getImagePaths(inputSource string) ([]string, error) {
	var imagePaths []string

	// 优先判断是否是.txt文件（解决os.Stat失败后仍尝试读取的问题）
	if strings.HasSuffix(strings.ToLower(inputSource), ".txt") {
		// 使用bufio.Scanner读取行，兼容不同系统换行符（\n/\r\n）
		file, err := os.Open(inputSource)
		if err != nil {
			return nil, fmt.Errorf("打开文本文件失败: %v", err)
		}
		defer file.Close() // 确保文件句柄关闭

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				// 可选：验证文本文件中的路径是否存在
				if _, err := os.Stat(line); err != nil {
					fmt.Printf("警告：文本文件中的路径 %s 不存在，已跳过\n", line)
					continue
				}
				imagePaths = append(imagePaths, line)
			}
		}

		// 检查scanner是否出错
		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("读取文本文件内容失败: %v", err)
		}
		return imagePaths, nil
	}

	// 检查输入源是否存在（非.txt文件）
	fileInfo, err := os.Stat(inputSource)
	if err != nil {
		return nil, fmt.Errorf("输入源不存在: %v", err)
	}

	if fileInfo.IsDir() {
		// 输入源是目录，遍历一级目录中的图像文件
		entries, err := os.ReadDir(inputSource)
		if err != nil {
			return nil, fmt.Errorf("读取目录出错: %v", err)
		}

		for _, entry := range entries {
			if entry.IsDir() {
				continue // 跳过子目录（如需递归，可在此处添加递归调用）
			}

			filePath := filepath.Join(inputSource, entry.Name())
			ext := strings.ToLower(filepath.Ext(entry.Name()))

			if supportedImageExts[ext] {
				imagePaths = append(imagePaths, filePath)
			} else if supportedVideoExts[ext] {
				// 视频文件提示并跳过，明确告知调用方
				fmt.Printf("提示：视频文件 %s 暂不支持，已跳过（功能待实现）\n", filePath)
			}
		}
	} else {
		// 输入源是单个文件
		ext := strings.ToLower(filepath.Ext(inputSource))

		if supportedImageExts[ext] {
			imagePaths = append(imagePaths, inputSource)
		} else if supportedVideoExts[ext] {
			// 视频文件明确返回警告（非错误），避免调用方误解
			fmt.Printf("提示：视频文件 %s 暂不支持（功能待实现）\n", inputSource)
		} else {
			return nil, fmt.Errorf("不支持的文件类型: %s（仅支持%v图像格式和%v视频格式）",
				ext, getKeys(supportedImageExts), getKeys(supportedVideoExts))
		}
	}

	return imagePaths, nil
}

// 辅助函数：获取map的key列表（用于友好提示）
func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// 计算颜色亮度的函数
// 用于判断背景颜色深浅，从而选择合适的文本颜色
=======
	fmt.Printf("使用参数: conf=%.2f, iou=%.2f, size=%d, text-location=%s\n",
		*confidenceThreshold, *iouThreshold, *modelInputSize, *systemTextLocation)
	os.Exit(run())
}

// 新增：计算颜色亮度的函数
>>>>>>> Stashed changes
func getLuminance(c color.RGBA) float64 {
	return 0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B)
}

// 新增：获取高对比度文本颜色
<<<<<<< Updated upstream
// 根据背景颜色自动选择黑色或白色文本，确保可读性
=======
>>>>>>> Stashed changes
func getContrastTextColor(backgroundColor color.RGBA) color.RGBA {
	luminance := getLuminance(backgroundColor)
	if luminance > 128 {
		return color.RGBA{0, 0, 0, 255} // 深色文本（黑色）
	}
	return color.RGBA{255, 255, 255, 255} // 浅色文本（白色）
}

<<<<<<< Updated upstream
// 检查字符串是否在数组中
// 用于过滤特定类别的检测结果
func checkStrIsInArray(str string, arr []string) bool {
	for _, item := range arr {
		if item == str {
			return true
		}
	}
	return false
}

// 处理独立图片目录的函数
func ProcessImageDirectory(inputDir, outputDir string) error {
	// 检查输入目录是否存在
	if _, err := os.Stat(inputDir); os.IsNotExist(err) {
		return fmt.Errorf("输入目录不存在: %v", err)
	}

	// 创建输出目录
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		err = os.MkdirAll(outputDir, 0755)
		if err != nil {
			return fmt.Errorf("创建输出目录失败: %v", err)
		}
	}

	// 获取目录中的所有图像文件
	imagePaths, err := getImagePaths(inputDir)
	if err != nil {
		return fmt.Errorf("获取目录中图像路径失败: %v", err)
	}

	// 生成输出路径列表
	outputPaths := make([]string, len(imagePaths))
	for i, imagePath := range imagePaths {
		imgName := filepath.Base(imagePath)
		outputPaths[i] = filepath.Join(outputDir, "detected_"+imgName)
	}

	// 使用并发处理图像
	return ConcurrentBatchProcessImages(imagePaths, outputPaths)
}

// 写入日志文件
// 记录程序运行过程中的重要事件和错误信息
func writeLogFile(level, message string) {
	// 创建logs目录
	logDir := "./logs"
	if _, err := os.Stat(logDir); os.IsNotExist(err) {
		err = os.Mkdir(logDir, 0755)
		if err != nil {
			fmt.Printf("创建日志目录失败: %v\n", err)
			return
		}
	}

	// 生成日志文件名（按日期）
	logFileName := fmt.Sprintf("%s/log_%s.txt", logDir, time.Now().Format("2006-01-02"))

	// 打开或创建日志文件
	logFile, err := os.OpenFile(logFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("打开日志文件失败: %v\n", err)
		return
	}
	defer logFile.Close()

	// 写入日志内容
	logEntry := fmt.Sprintf("%s %s %s\n", time.Now().Format("2006-01-02 15:04:05"), level, message)
	_, err = logFile.WriteString(logEntry)
	if err != nil {
		fmt.Printf("写入日志失败: %v\n", err)
		return
	}
}

// 获取区域平均颜色（用于系统文本背景）
// 用于在不同背景上显示系统文本时提供合适的背景色
=======
// 新增：获取区域平均颜色（用于系统文本背景）
>>>>>>> Stashed changes
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

// 新增：绘制系统文本函数
<<<<<<< Updated upstream
// 在图像上添加系统标识文字，如监控系统名称等
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======

	fmt.Printf("系统文本: %s, 位置: %s, 坐标: (%d,%d)\n",
		text, location, textX, textY)
>>>>>>> Stashed changes
}

// initChineseFont 初始化中文字体
// 查找系统中可用的中文字体文件并加载
func initChineseFont() error {
	fontPaths := findfont.List()
	var fontPath string

	// 常见的中文字体文件名
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
// 释放字体占用的内存资源
func cleanupFont() {
	if chineseFont != nil {
		chineseFont.Close()
	}
}

// getChineseLabel 获取中文标签
// 将英文标签转换为对应的中文标签
func getChineseLabel(englishLabel string) string {
	if chinese, exists := detectLabeltMap[englishLabel]; exists {
		return chinese
	}
	return englishLabel
}

<<<<<<< Updated upstream
// 图片检测输出结果 输入图片地址 输出检测结果中的对象描述:对象个数;描述:对象1是*,置信度;错误信息
// 核心检测函数，执行完整的检测流程
func detectImage(inputImagePath, outputImagePath string) (int, string, error) {
	os.Setenv("LC_ALL", "zh_CN.UTF-8")
=======
func run() int {
	// 初始化中文字体
>>>>>>> Stashed changes
	if err := initChineseFont(); err != nil {
		fmt.Printf("警告: 中文字体初始化失败: %v\n", err)
	} else {
		defer cleanupFont()
	}

	originalPic, e := loadImageFile(inputImagePath)
	if e != nil {
		return 0, "", e
	}
	originalWidth := originalPic.Bounds().Dx()
	originalHeight := originalPic.Bounds().Dy()

	modelSession, e := initSession()
	if e != nil {
		return 0, "", e
	}
	defer modelSession.Destroy()

	var allBoxes []boundingBox

	if *useAugment {
		// 原图
		scaleInfo, e := prepareInput(originalPic, modelSession.Input)
		if e != nil {
			return 0, "", e
		}
		modelSession.Session.Run()
		originalBoxes := processOutput(modelSession.Output.GetData(), originalWidth, originalHeight,
			float32(*confidenceThreshold), float32(*iouThreshold), scaleInfo)
		allBoxes = append(allBoxes, originalBoxes...)

		// 水平翻转图像
		flippedPic := flipHorizontal(originalPic)
		scaleInfo, e = prepareInput(flippedPic, modelSession.Input)
		if e == nil {
			modelSession.Session.Run()
			flippedBoxes := processOutput(modelSession.Output.GetData(), originalWidth, originalHeight,
				float32(*confidenceThreshold), float32(*iouThreshold), scaleInfo)
			for i := range flippedBoxes {
				flippedBoxes[i] = flipBoundingBox(flippedBoxes[i], originalWidth)
			}
			allBoxes = append(allBoxes, flippedBoxes...)
		}

		// 合并框并 NMS
		if len(allBoxes) > 0 {
			allBoxes = nonMaxSuppression(allBoxes, float32(*iouThreshold))
		}
	} else {
		scaleInfo, e := prepareInput(originalPic, modelSession.Input)
		if e != nil {
			return 0, "", e
		}
		modelSession.Session.Run()
		allBoxes = processOutput(modelSession.Output.GetData(), originalWidth, originalHeight,
			float32(*confidenceThreshold), float32(*iouThreshold), scaleInfo)
	}

	var outObjectStr string
	var num int
	for _, box := range allBoxes {
		if checkStrIsInArray(box.label, []string{"person", "car", "motorcycle", "bus", "truck"}) {
			num++
			chineseLabel := getChineseLabel(box.label)
			confStr := fmt.Sprintf("%.2f", float32(math.Round(float64(box.confidence*100))/100))
			outObjectStr += "对象" + strconv.Itoa(num) + ": " + box.label + chineseLabel + ", 置信度: " + confStr + " ; "
		}
	}
	if num > 0 {
		outObjectStr = " AI分析到危险对象共有 " + strconv.Itoa(num) + " 个, " + outObjectStr
	} else {
		outObjectStr = "未检测到危险对象"
	}

	e = drawBoundingBoxesWithLabels(originalPic, allBoxes, outputImagePath)
	if e != nil {
		return num, outObjectStr, e
	}

	return num, outObjectStr, nil
}

<<<<<<< Updated upstream
// 安全的ONNX Runtime环境初始化函数
// 确保ONNX Runtime只被初始化一次，保证线程安全

func initializeORTEnvironment() error {
	ortInitMutex.Lock()
	defer ortInitMutex.Unlock()
	if ortInitialized {
		return nil
=======
// ensureDirForFile 确保给定文件路径的父目录存在，若不存在则创建
func ensureDirForFile(filePath string) error {
	dir := filepath.Dir(filePath)

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err = os.MkdirAll(dir, 0755)
		if err != nil {
			return fmt.Errorf("无法创建目录 %s: %w", dir, err)
		}
		fmt.Printf("✅ 目录已创建: %s\n", dir)
	} else if err != nil {
		return fmt.Errorf("检查目录状态失败 %s: %w", dir, err)
	} else {
		fmt.Printf("📁 目录已存在: %s\n", dir)
>>>>>>> Stashed changes
	}
	libPath := getSharedLibPath()
	if libPath == "" {
		return errors.New("未找到ONNX Runtime库")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("初始化ORT环境失败: %w", err)
	}
	ortInitialized = true
	return nil
}

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

// boundingBox 表示检测到的目标的边界框
// 存储检测结果的位置、类别和置信度信息
type boundingBox struct {
	label      string  // 检测到的对象类别标签
	confidence float32 // 检测置信度（0-1之间）
	x1, y1     float32 // 边界框左上角坐标
	x2, y2     float32 // 边界框右下角坐标
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

// 加载图像文件
// 支持多种图像格式（JPEG、PNG、GIF等）
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

<<<<<<< Updated upstream
// 旧函数已被替换，请使用resizeWithLetterbox函数

// LetterBox类的rect=False模式实现（auto=False）
// 对应Python中LetterBox的auto=False参数，用于rect=False模式（标准letterbox）
// 保持长宽比，将图像缩放到最短边等于目标尺寸，用灰色填充
func resizeWithLetterbox(img image.Image, targetSize int) (image.Image, ScaleInfo) {
=======
// resizeWithAspectRatio 保持长宽比的缩放（rect=True的功能）
func resizeWithAspectRatio(img image.Image, targetWidth, targetHeight int) image.Image {
>>>>>>> Stashed changes
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

<<<<<<< Updated upstream
	// 计算缩放比例，保持长宽比，确保最短边适应目标尺寸
	scale := float64(targetSize) / math.Max(float64(originalWidth), float64(originalHeight))
	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	// 缩放图像
	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)
	result := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))

	// 填充灰色背景 (114, 114, 114) - YOLO标准
	grayFill := &image.Uniform{color.RGBA{114, 114, 114, 255}}
	draw.Draw(result, result.Bounds(), grayFill, image.Point{}, draw.Src)

	// 将缩放后的图像居中放置
	offsetX := (targetSize - newWidth) / 2
	offsetY := (targetSize - newHeight) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+newWidth, offsetY+newHeight),
		resized, image.Point{}, draw.Src)

	// 计算实际的缩放比例（相对于原始图像）
	scaleX := float32(newWidth) / float32(originalWidth)
	scaleY := float32(newHeight) / float32(originalHeight)

	scaleInfo := ScaleInfo{
		ScaleX:    scaleX,
		ScaleY:    scaleY,
		PadLeft:   offsetX,
		PadTop:    offsetY,
		NewWidth:  newWidth,
		NewHeight: newHeight,
=======
	// 计算缩放比例，保持长宽比
	scale := float64(targetWidth) / float64(originalWidth)
	if float64(targetHeight)/float64(originalHeight) < scale {
		scale = float64(targetHeight) / float64(originalHeight)
>>>>>>> Stashed changes
	}

	return result, scaleInfo
}

// LetterBox类的rect=True模式实现（auto=True）
// 对应Python中LetterBox的auto=True参数，用于rect=True模式
// 保持长宽比，同时确保尺寸能被步长(stride)整除，以提高批处理效率
func resizeWithRectScaling(img image.Image, targetSize int) (image.Image, ScaleInfo) {
	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	scale := float64(targetSize) / math.Min(float64(originalWidth), float64(originalHeight))
	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	resized := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Bilinear)

<<<<<<< Updated upstream
	// 中心裁剪成 640x640
	startX := (newWidth - targetSize) / 2
	startY := (newHeight - targetSize) / 2
	if startX < 0 {
		startX = 0
	}
	if startY < 0 {
		startY = 0
	}

	cropped := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.Draw(cropped, cropped.Bounds(), resized, image.Point{startX, startY}, draw.Src)
=======
	// 填充灰色背景（YOLO标准做法）
	draw.Draw(result, result.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)

	// 将缩放后的图像居中放置
	offsetX := (targetWidth - newWidth) / 2
	offsetY := (targetHeight - newHeight) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+newWidth, offsetY+newHeight),
		resized, image.Point{}, draw.Src)
>>>>>>> Stashed changes

	scaleX := float32(newWidth) / float32(originalWidth)
	scaleY := float32(newHeight) / float32(originalHeight)

	scaleInfo := ScaleInfo{
		ScaleX:    scaleX,
		ScaleY:    scaleY,
		PadLeft:   startX,
		PadTop:    startY,
		NewWidth:  newWidth,
		NewHeight: newHeight,
	}
	return cropped, scaleInfo
}

// 获取ONNX Runtime共享库路径
// 根据不同的操作系统和架构返回相应的动态库文件路径
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
<<<<<<< Updated upstream
	return ""
=======
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
	return nil
>>>>>>> Stashed changes
}

// 初始化ONNX Runtime会话
// 创建模型推理所需的会话和张量
func initSession() (*ModelSession, error) {
<<<<<<< Updated upstream
=======
	// 先初始化ONNX Runtime环境
>>>>>>> Stashed changes
	if err := initializeORTEnvironment(); err != nil {
		return nil, err
	}
	size := *modelInputSize
	inputShape := ort.NewShape(int64(*batchSize), 3, int64(size), int64(size))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %w", err)
	}
	outputShape := ort.NewShape(int64(*batchSize), 84, 8400) // YOLO 输出
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
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)
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

// 处理模型输出
// 解析模型输出的原始数据，提取边界框、类别和置信度信息
func processOutput(output []float32, originalWidth, originalHeight int, confThreshold, iouThresh float32, scaleInfo ScaleInfo) []boundingBox {
	boundingBoxes := make([]boundingBox, 0, 8400)

<<<<<<< Updated upstream
	numAnchors := 8400
	numClasses := 80

	scaleX := scaleInfo.ScaleX
	scaleY := scaleInfo.ScaleY
=======
	// 根据是否使用矩形缩放来计算不同的缩放参数
	var scaleX, scaleY float32
	var padX, padY int

	if *useRectScaling {
		// 矩形缩放：保持长宽比
		scaleUsed := float32(640.0) / float32(originalWidth)
		scaleH := float32(640.0) / float32(originalHeight)
		if scaleH < scaleUsed {
			scaleUsed = scaleH
		}

		newWidth := int(float32(originalWidth) * scaleUsed)
		newHeight := int(float32(originalHeight) * scaleUsed)
		padX = (640 - newWidth) / 2
		padY = (640 - newHeight) / 2

		scaleX = scaleUsed
		scaleY = scaleUsed
	} else {
		// 非矩形缩放：直接拉伸，分别计算宽度和高度的缩放比例
		scaleX = float32(640.0) / float32(originalWidth)
		scaleY = float32(640.0) / float32(originalHeight)
		padX = 0
		padY = 0
	}
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
		// 映射回原图坐标
		origCenterX := (xc - float32(scaleInfo.PadLeft)) / scaleX
		origCenterY := (yc - float32(scaleInfo.PadTop)) / scaleY
		origW := w / scaleX
		origH := h / scaleY

		x1 := origCenterX - origW/2
		y1 := origCenterY - origH/2
		x2 := origCenterX + origW/2
		y2 := origCenterY + origH/2
=======
		// 统一的坐标转换公式，根据缩放模式自动处理
		x1 := (xc - w/2 - float32(padX)) / scaleX
		y1 := (yc - h/2 - float32(padY)) / scaleY
		x2 := (xc + w/2 - float32(padX)) / scaleX
		y2 := (yc + h/2 - float32(padY)) / scaleY
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
			label:      englishLabel,
			confidence: finalConf,
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
=======
			label:      chineseLabel,
			confidence: maxProb,
			x1:         x1, y1: y1, x2: x2, y2: y2,
>>>>>>> Stashed changes
		})
	}

	// 按置信度排序
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	result := nonMaxSuppression(boundingBoxes, iouThresh)
	return result
}

<<<<<<< Updated upstream
// 准备输入数据
// 将图像数据转换为模型输入所需的格式（归一化RGB张量）
func prepareInput(pic image.Image, dst *ort.Tensor[float32]) (ScaleInfo, error) {
	inputSize := *modelInputSize
	channelSize := inputSize * inputSize
	data := dst.GetData()
	if len(data) < 3*channelSize {
		return ScaleInfo{}, errors.New("输入张量长度不足")
	}
	var resizedImg image.Image
	var scaleInfo ScaleInfo
	if *useRectScaling {
		resizedImg, scaleInfo = resizeWithRectScaling(pic, inputSize)
	} else {
		resizedImg, scaleInfo = resizeWithLetterbox(pic, inputSize)
	}
	// TTA 修正: 对齐框和对象
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
	return scaleInfo, nil
}

// 确保值在指定范围内
=======
// 同时需要修正 prepareInput 函数中的缩放逻辑
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

	var resizedImg image.Image
	if *useRectScaling {
		// 矩形缩放：保持长宽比
		resizedImg = resizeWithAspectRatio(pic, inputSize, inputSize)
	} else {
		// 非矩形缩放：直接拉伸到目标尺寸
		resizedImg = resize.Resize(uint(inputSize), uint(inputSize), pic, resize.Bilinear)
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
	return nil
}

// 确保 clamp 函数存在
>>>>>>> Stashed changes
func clamp(value, min, max float32) float32 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

<<<<<<< Updated upstream
// min和max辅助函数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 确保至少有一个工作协程
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 水平翻转图像
// 用于测试时增强(TTA)，提高检测精度
func flipHorizontal(img image.Image) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	result := image.NewRGBA(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			result.Set(w-x-1, y, img.At(x, y))
		}
	}
	return result
}

// 旋转图像（简单实现，仅支持90度倍数旋转）
// 预留功能，可用于更多数据增强方法
func rotateImage(img image.Image, degrees int) image.Image {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	switch degrees {
	case 90:
		result := image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				result.Set(y, w-x-1, img.At(x, y))
			}
		}
		return result
	case 180:
		result := image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				result.Set(w-x-1, h-y-1, img.At(x, y))
			}
		}
		return result
	case 270:
		result := image.NewRGBA(image.Rect(0, 0, h, w))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				result.Set(h-y-1, x, img.At(x, y))
			}
		}
		return result
	default:
		// 角度不为90度倍数时，返回原始图像
		return img
	}
}

// 水平翻转边界框（用于TTA结果融合）
// 将翻转图像上的检测结果转换回原始图像坐标
func flipBoundingBox(box boundingBox, imageWidth int) boundingBox {
	// 水平翻转x坐标
	originalX1 := box.x1
	originalX2 := box.x2
	box.x1 = float32(imageWidth) - originalX2
	box.x2 = float32(imageWidth) - originalX1
	return box
}

// 非极大值抑制(NMS)
// 去除重复的检测框，保留置信度最高的框
=======
>>>>>>> Stashed changes
func nonMaxSuppression(boxes []boundingBox, iouThreshold float32) []boundingBox {
	if len(boxes) == 0 {
		return boxes
	}

	// 按置信度降序排序
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].confidence > boxes[j].confidence
	})

	selected := make([]boundingBox, 0)
	picked := make([]bool, len(boxes))

	// 按类别分组进行NMS抑制 - 仿照官方Python的batched_nms实现
	for i := 0; i < len(boxes); i++ {
		if picked[i] {
			continue
		}

		selected = append(selected, boxes[i])
		picked[i] = true

		// 只对相同类别的框进行NMS抑制
		for j := i + 1; j < len(boxes); j++ {
			if picked[j] || boxes[i].label != boxes[j].label {
				continue
			}

			// 计算IoU
			iou := boxes[i].iou(&boxes[j])
			if iou >= iouThreshold { // 使用 >= 与官方Python代码保持一致
				picked[j] = true
			}
		}
	}
	return selected
}

<<<<<<< Updated upstream
// 绘制边界框和标签
// 在原图上绘制检测结果，包括边界框、标签和置信度
=======
// 修改drawBoundingBoxesWithLabels函数，添加系统文本绘制
>>>>>>> Stashed changes
func drawBoundingBoxesWithLabels(img image.Image, boxes []boundingBox, outputPath string) error {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, image.Point{}, draw.Src)

<<<<<<< Updated upstream
	// 定义不同类别的颜色映射 - 使用更鲜明的颜色
	var colors = map[string]color.RGBA{
		"person":         {0, 0, 255, 255},     // 纯红色 - 人物
		"bicycle":        {255, 165, 0, 255},   // 橙色 - 自行车
		"car":            {0, 255, 0, 255},     // 纯绿色 - 汽车
		"motorcycle":     {255, 255, 0, 255},   // 纯黄色 - 摩托车
		"airplane":       {255, 0, 255, 255},   // 洋红色 - 飞机
		"bus":            {0, 255, 255, 255},   // 青色 - 巴士
		"train":          {128, 0, 128, 255},   // 紫色 - 火车
		"truck":          {0, 0, 255, 255},     // 纯蓝色 - 卡车
		"boat":           {0, 128, 255, 255},   // 深天蓝色 - 船
		"traffic light":  {128, 0, 128, 255},   // 紫色 - 红绿灯
		"fire hydrant":   {0, 0, 139, 255},     // 深蓝色 - 消防栓
		"stop sign":      {255, 20, 147, 255},  // 深粉色 - 停车标志
		"parking meter":  {218, 165, 32, 255},  // 金色 - 停车计时器
		"bench":          {139, 69, 19, 255},   // 巧克力色 - 长凳
		"bird":           {238, 130, 238, 255}, // 紫罗兰色 - 鸟
		"cat":            {255, 192, 203, 255}, // 粉色 - 猫
		"dog":            {123, 104, 238, 255}, // 中紫色 - 狗
		"horse":          {255, 69, 0, 255},    // 橙红色 - 马
		"sheep":          {144, 238, 144, 255}, // 浅绿色 - 羊
		"cow":            {240, 230, 140, 255}, // 亚麻色 - 牛
		"elephant":       {128, 128, 0, 255},   // 橄榄色 - 大象
		"bear":           {165, 42, 42, 255},   // 棕色 - 熊
		"zebra":          {255, 255, 255, 255}, // 白色 - 斑马
		"giraffe":        {255, 228, 181, 255}, // 蜜蜂色 - 长颈鹿
		"backpack":       {70, 130, 180, 255},  // 钢蓝色 - 背包
		"umbrella":       {255, 193, 37, 255},  // 金菊色 - 雨伞
		"handbag":        {220, 20, 60, 255},   // 猩红色 - 手提包
		"tie":            {75, 0, 130, 255},    // 深紫色 - 领带
		"suitcase":       {244, 164, 96, 255},  // 沙棕色 - 行李箱
		"frisbee":        {50, 205, 50, 255},   // 石灰绿 - 飞盘
		"skis":           {176, 224, 230, 255}, // 粉蓝色 - 滑雪板
		"snowboard":      {106, 90, 205, 255},  // 紫罗兰色 - 雪板
		"sports ball":    {255, 140, 0, 255},   // 深橙色 - 运动球
		"kite":           {148, 0, 211, 255},   // 深紫色 - 风筝
		"baseball bat":   {165, 42, 42, 255},   // 棕色 - 棒球棍
		"baseball glove": {255, 20, 147, 255},  // 深粉色 - 棒球手套
		"skateboard":     {30, 144, 255, 255},  // 道奇蓝 - 滑板
		"surfboard":      {255, 105, 180, 255}, // 粉红色 - 冲浪板
		"tennis racket":  {0, 255, 127, 255},   // 草绿色 - 网球拍
		"bottle":         {216, 191, 216, 255}, // 薄荷奶油色 - 瓶子
		"wine glass":     {255, 218, 185, 255}, // 桃色 - 酒杯
		"cup":            {255, 182, 193, 255}, // 浅粉色 - 杯子
		"fork":           {112, 128, 144, 255}, // 石板灰 - 叉子
		"knife":          {178, 34, 34, 255},   // 鲜红色 - 刀
		"spoon":          {220, 220, 220, 255}, // 浅灰色 - 勺子
		"bowl":           {255, 222, 173, 255}, // 蜂蜡色 - 碗
		"banana":         {255, 255, 0, 255},   // 纯黄色 - 香蕉
		"apple":          {255, 99, 71, 255},   // 番茄红 - 苹果
		"sandwich":       {184, 134, 11, 255},  // 深卡其色 - 三明治
		"orange":         {255, 165, 0, 255},   // 纯橙色 - 橙子
		"broccoli":       {34, 139, 34, 255},   // 森林绿 - 西兰花
		"carrot":         {255, 140, 0, 255},   // 深橙色 - 胡萝卜
		"hot dog":        {188, 143, 143, 255}, // 石色 - 热狗
		"pizza":          {205, 133, 63, 255},  // 石褐色 - 披萨
		"donut":          {139, 69, 19, 255},   // 巧克力色 - 甜甜圈
		"cake":           {255, 192, 203, 255}, // 粉色 - 蛋糕
		"chair":          {107, 142, 35, 255},  // 黄橄榄绿 - 椅子
		"couch":          {47, 79, 79, 255},    // 暗瓦灰色 - 沙发
		"potted plant":   {34, 139, 34, 255},   // 森林绿 - 盆栽
		"bed":            {255, 105, 180, 255}, // 粉红色 - 床
		"dining table":   {210, 105, 30, 255},  // 巧克力色 - 餐桌
		"toilet":         {175, 238, 238, 255}, // 浅碧绿色 - 厕所
		"tv":             {0, 191, 255, 255},   // 深天蓝色 - 电视
		"laptop":         {95, 158, 160, 255},  // 青铜色 - 笔记本电脑
		"mouse":          {221, 160, 221, 255}, // 蓟色 - 鼠标
		"remote":         {138, 43, 226, 255},  // 蓝紫色 - 遥控器
		"keyboard":       {112, 128, 144, 255}, // 石板灰 - 键盘
		"cell phone":     {219, 112, 147, 255}, // 苍紫罗兰色 - 手机
		"microwave":      {186, 85, 211, 255},  // 紫色 - 微波炉
		"oven":           {139, 0, 0, 255},     // 暗红色 - 烤箱
		"toaster":        {160, 82, 45, 255},   // 木色 - 烤面包机
		"sink":           {0, 139, 139, 255},   // 深青色 - 水槽
		"refrigerator":   {70, 130, 180, 255},  // 钢蓝色 - 冰箱
		"book":           {160, 32, 240, 255},  // 紫色 - 书
		"clock":          {255, 215, 0, 255},   // 金色 - 钟
		"vase":           {216, 191, 216, 255}, // 薄荷奶油色 - 花瓶
		"scissors":       {128, 128, 0, 255},   // 橄榄色 - 剪刀
		"teddy bear":     {210, 105, 30, 255},  // 巧克力色 - 泰迪熊
		"hair drier":     {221, 160, 221, 255}, // 蓟色 - 吹风机
		"toothbrush":     {255, 182, 193, 255}, // 浅粉色 - 牙刷
		"default":        {128, 128, 128, 255}, // 默认颜色(灰色)
=======
	// 定义不同类别的颜色映射
	colors := map[string]color.RGBA{
		"人员":      {255, 0, 0, 255},     // 红色
		"汽车":      {0, 255, 0, 255},     // 绿色
		"巴士":      {0, 0, 255, 255},     // 蓝色
		"摩托车":     {255, 255, 0, 255},   // 黄色
		"卡车":      {255, 0, 255, 255},   // 紫色
		"自行车":     {0, 255, 255, 255},   // 青色
		"default": {128, 128, 128, 255}, // 灰色(默认)
>>>>>>> Stashed changes
	}

	// 绘制每个检测框
	for _, box := range boxes {
		boxColor, exists := colors[box.label]
		if !exists {
			boxColor = colors["default"]
		}

		// 绘制边界框
		for y := int(box.y1); y <= int(box.y2); y++ {
			if y < 0 || y >= bounds.Dy() {
				continue
			}
			// 左右两条竖线
			if int(box.x1) >= 0 && int(box.x1) < bounds.Dx() {
				rgba.Set(int(box.x1), y, boxColor)
			}
			if int(box.x2) >= 0 && int(box.x2) < bounds.Dx() {
				rgba.Set(int(box.x2), y, boxColor)
			}
		}
		for x := int(box.x1); x <= int(box.x2); x++ {
			if x < 0 || x >= bounds.Dx() {
				continue
			}
			// 上下两条横线
			if int(box.y1) >= 0 && int(box.y1) < bounds.Dy() {
				rgba.Set(x, int(box.y1), boxColor)
			}
			if int(box.y2) >= 0 && int(box.y2) < bounds.Dy() {
				rgba.Set(x, int(box.y2), boxColor)
			}
		}

		// 使用改进的drawLabel函数，使用框颜色作为背景色，确保文本与背景对比度
		drawLabel(rgba, box, boxColor)
	}

<<<<<<< Updated upstream
	// 绘制系统文本
	drawSystemText(rgba, *systemTextLocation)

	// 保存图像
=======
	// 新增：绘制系统文本
	drawSystemText(rgba, *systemTextLocation)

>>>>>>> Stashed changes
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("创建输出文件失败: %w", err)
	}
	defer outFile.Close()

	err = jpeg.Encode(outFile, rgba, &jpeg.Options{Quality: 90})
	if err != nil {
		return fmt.Errorf("编码输出图像失败: %w", err)
	}

	return nil
}

// 测量文本宽度和高度的辅助函数
// 计算文本在指定字体下的尺寸
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

<<<<<<< Updated upstream
// 修改后的drawLabel函数，支持中文标签
// 在边界框旁边绘制类别标签和置信度
=======
// 修改drawLabel函数，改进文本颜色对比度
>>>>>>> Stashed changes
func drawLabel(img *image.RGBA, box boundingBox, boxColor color.RGBA) {
	chineseLabel := getChineseLabel(box.label)
	labelText := fmt.Sprintf("%s(%.2f)", chineseLabel, box.confidence) // 与drawBoundingBoxesWithLabels中的格式一致
	rect := box.toRect()

	textWidth, textHeight := measureText(labelText, chineseFont)

	// 计算标签文本位置，确保在图像边界内
	textX := rect.Min.X + 5
	textY := rect.Min.Y - 5

<<<<<<< Updated upstream
=======
	// 边界检查和位置调整（原有逻辑保持不变）
>>>>>>> Stashed changes
	imgHeight := img.Bounds().Dy()
	if textY < textHeight {
		textY = rect.Min.Y + textHeight + 5
	}
	if textY > imgHeight-5 {
		textY = rect.Min.Y - textHeight - 5
<<<<<<< Updated upstream
		if textY < 5 {
=======
		if textY < textHeight {
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
	// 计算标签背景矩形
=======
	// 调整背景色块大小
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
	// 使用框颜色作为背景色，确保框和标签底色一致
	// 并使用高对比度文本颜色
	textColor := getContrastTextColor(boxColor)

	// 绘制标签背景和文本
	drawTextBackground(img, bgX, bgY, bgWidth, bgHeight, boxColor) // 使用框颜色作为背景
	drawText(img, textX, textY, labelText, textColor)              // 使用对比色文本
=======
	// 修改：使用高对比度文本颜色
	textColor := getContrastTextColor(boxColor)

	// 绘制背景色块
	drawTextBackground(img, bgX, bgY, bgWidth, bgHeight, boxColor)

	// 绘制文本（使用对比色）
	drawText(img, textX, textY, labelText, textColor)

	fmt.Printf("文本: %s, 颜色对比度已优化, 位置: (%d,%d)\n", labelText, textX, textY)
>>>>>>> Stashed changes
}

// 改进的drawTextBackground函数
// 绘制标签文本的背景矩形
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

	// 绘制背景矩形
	for i := x; i < x+width && i < img.Bounds().Dx(); i++ {
		for j := y; j < y+height && j < img.Bounds().Dy(); j++ {
			img.Set(i, j, bgColor)
		}
	}
}

// 修改后的drawText函数，支持中文显示
// 在图像上绘制文本，优先使用中文字体
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

<<<<<<< Updated upstream
// YOLO类别标签（英文原始标签）[1,2](@ref)
// YOLOv8模型支持的80个类别
=======
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

// YOLO类别标签（英文原始标签）
>>>>>>> Stashed changes
var yoloClasses = []string{
	"3D CG rendering", "3D glasses", "abacus", "abalone", "monastery", "belly",
}

<<<<<<< Updated upstream
// 中英标签映射
// 将YOLO英文标签映射为中文标签
=======
// 中英文标签映射表
>>>>>>> Stashed changes
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
	"traffic light":  "红绿灯",
	"fire hydrant":   "消防栓",
	"stop sign":      "停车标志",
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
	"snowboard":      "雪板",
	"sports ball":    "运动球",
	"kite":           "风筝",
	"baseball bat":   "棒球棍",
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
	"potted plant":   "盆栽",
	"bed":            "床",
	"dining table":   "餐桌",
	"toilet":         "厕所",
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
	"clock":          "钟",
	"vase":           "花瓶",
	"scissors":       "剪刀",
	"teddy bear":     "泰迪熊",
	"hair drier":     "吹风机",
	"toothbrush":     "牙刷",
}

// 根据原始颜色计算高对比度背景颜色
// 如果原始颜色太亮，则使用深色背景；如果太暗，则使用浅色背景
func getHighContrastBackgroundColor(originalColor color.RGBA) color.RGBA {
	luminance := getLuminance(originalColor)

	// 如果原始颜色很亮（亮度值大于128），使用深色背景
	if luminance > 128 {
		// 返回半透明黑色背景，这样可以保留一些原始颜色的影响
		return color.RGBA{R: originalColor.R / 3, G: originalColor.G / 3, B: originalColor.B / 3, A: 200}
	} else {
		// 如果原始颜色较暗，使用浅色背景
		// 确保背景足够亮以提供对比度
		avg := (uint32(originalColor.R) + uint32(originalColor.G) + uint32(originalColor.B)) / 3
		increase := uint8(180 - avg)
		if increase > 0 {
			r := originalColor.R + increase
			if r < originalColor.R { // 溢出检查
				r = 255
			}
			g := originalColor.G + increase
			if g < originalColor.G { // 溢出检查
				g = 255
			}
			b := originalColor.B + increase
			if b < originalColor.B { // 溢出检查
				b = 255
			}
			return color.RGBA{R: r, G: g, B: b, A: 220}
		} else {
			return color.RGBA{R: 200, G: 200, B: 200, A: 220}
		}
	}
}
