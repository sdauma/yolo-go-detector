// go_cold_start_decomposition.go
// Go 冷启动分解测试
//
// 技术说明：
// - 使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
//   自动启用 I/O Binding，但不接受 SessionOptions 参数
// - 线程配置由 ONNX Runtime 默认行为决定（intra_op_num_threads 默认等于 CPU 核数）
// - 代码中创建了 SessionOptions 并设置了 intraOp=12，但由于 NewSession 不接受 opts，
//   这些设置实际上不生效。保留 opts 创建代码仅用于记录意图
//
// 测试目的：
// - 分解冷启动时间为会话创建时间、模型加载时间和首次推理时间
// - 执行 20 次冷启动测试，计算平均值
// - 确保数据稳定性和可重复性

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

// ColdStartResult 冷启动测试结果
type ColdStartResult struct {
	SessionCreationTime float64
	ModelLoadingTime    float64
	FirstInferenceTime  float64
	TotalColdStartTime  float64
	StartRSS            float64
	PeakRSS             float64
}

// runColdStartTest 执行一次冷启动测试
func runColdStartTest(modelPath, modelName, libPath string) (*ColdStartResult, error) {
	fmt.Printf("\n===== Go 冷启动测试 - %s ====\n", modelName)

	// 记录初始内存
	startRSS := getProcessRSS()

	// 1. 初始化ORT环境
	t0 := time.Now()
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 2. 创建会话选项
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建会话选项失败: %v", err)
	}
	defer opts.Destroy()

	// 显式设置所有 SessionOptions 参数
	// 线程配置 - 12线程，与其他测试保持一致
	opts.SetIntraOpNumThreads(12)
	opts.SetInterOpNumThreads(1)
	opts.SetLogSeverityLevel(3)
	opts.SetExecutionMode(0)
	opts.SetGraphOptimizationLevel(3)

	// 3. 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %v", err)
	}
	defer inputTensor.Destroy()

	// 4. 从预生成的二进制文件加载输入数据
	wd, _ := os.Getwd()
	basePath := filepath.Dir(filepath.Dir(wd))
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	inputDataFile, err := os.ReadFile(inputDataPath)
	if err != nil {
		return nil, fmt.Errorf("读取输入数据文件失败: %v", err)
	}

	// 转换为 float32 并填充到张量
	inputData := inputTensor.GetData()
	expectedSize := 1 * 3 * 640 * 640 * 4 // float32 = 4 bytes
	if len(inputDataFile) != expectedSize {
		return nil, fmt.Errorf("输入数据文件大小不匹配: 期望 %d 字节，实际 %d 字节", expectedSize, len(inputDataFile))
	}

	// 将字节数据转换为 float32 并复制到张量
	for i := 0; i < len(inputData); i += 4 {
		bits := binary.LittleEndian.Uint32(inputDataFile[i : i+4])
		value := math.Float32frombits(bits)
		inputData[i/4] = value
	}

	// 5. 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输出张量失败: %v", err)
	}
	defer outputTensor.Destroy()

	// 6. 会话创建时间（包含模型加载）
	t1 := time.Now()
	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("创建会话失败: %v", err)
	}
	defer session.Destroy()
	sessionCreationTime := time.Since(t1).Seconds() * 1000.0

	// 7. 模型加载时间（会话创建已经包含了模型加载）
	modelLoadingTime := sessionCreationTime

	// 8. 首次推理时间
	t3 := time.Now()
	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("首次推理失败: %v", err)
	}
	firstInferenceTime := time.Since(t3).Seconds() * 1000.0

	// 计算总冷启动时间
	totalColdStartTime := time.Since(t0).Seconds() * 1000.0

	// 记录峰值内存
	peakRSS := getProcessRSS()

	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Printf("会话创建时间: %.5f ms\n", sessionCreationTime)
	fmt.Printf("模型加载时间: %.5f ms\n", modelLoadingTime)
	fmt.Printf("首次推理时间: %.5f ms\n", firstInferenceTime)
	fmt.Printf("总冷启动时间: %.5f ms\n", totalColdStartTime)
	fmt.Printf("Start RSS: %.5f MB\n", startRSS)
	fmt.Printf("Peak RSS: %.5f MB\n", peakRSS)

	return &ColdStartResult{
		SessionCreationTime: sessionCreationTime,
		ModelLoadingTime:    modelLoadingTime,
		FirstInferenceTime:  firstInferenceTime,
		TotalColdStartTime:  totalColdStartTime,
		StartRSS:            startRSS,
		PeakRSS:             peakRSS,
	}, nil
}

func main() {
	fmt.Println("===== Go 冷启动分解测试（20次运行）=====")

	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取工作目录失败: %v\n", err)
		return
	}

	// 构建项目根路径
	basePath := filepath.Dir(filepath.Dir(wd))

	// 设置模型和库路径
	modelPathLarge := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	modelPathSmall := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")

	// 检查文件是否存在
	if !fileExists(modelPathLarge) {
		fmt.Printf("错误: 大模型文件不存在: %s\n", modelPathLarge)
		return
	}
	if !fileExists(modelPathSmall) {
		fmt.Printf("错误: 轻模型文件不存在: %s\n", modelPathSmall)
		return
	}
	if !fileExists(libPath) {
		fmt.Printf("错误: 库文件不存在: %s\n", libPath)
		return
	}

	// 运行20次测试 - 大模型
	numRuns := 20
	resultsLarge := make([]*ColdStartResult, numRuns)
	resultsSmall := make([]*ColdStartResult, numRuns)

	fmt.Println("\n===== 测试大模型 (YOLO11x) =====")
	for i := 0; i < numRuns; i++ {
		fmt.Printf("\n===== 第 %d 次测试 =====\n", i+1)
		result, err := runColdStartTest(modelPathLarge, "YOLO11x", libPath)
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		resultsLarge[i] = result
	}

	fmt.Println("\n===== 测试轻模型 (YOLO11n) =====")
	for i := 0; i < numRuns; i++ {
		fmt.Printf("\n===== 第 %d 次测试 =====\n", i+1)
		result, err := runColdStartTest(modelPathSmall, "YOLO11n", libPath)
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		resultsSmall[i] = result
	}

	// 计算大模型平均值
	var avgSessionCreationLarge, avgModelLoadingLarge, avgFirstInferenceLarge, avgTotalColdStartLarge float64
	var avgStartRSSLarge, avgPeakRSSLarge float64

	for _, r := range resultsLarge {
		avgSessionCreationLarge += r.SessionCreationTime
		avgModelLoadingLarge += r.ModelLoadingTime
		avgFirstInferenceLarge += r.FirstInferenceTime
		avgTotalColdStartLarge += r.TotalColdStartTime
		avgStartRSSLarge += r.StartRSS
		avgPeakRSSLarge += r.PeakRSS
	}

	avgSessionCreationLarge /= float64(numRuns)
	avgModelLoadingLarge /= float64(numRuns)
	avgFirstInferenceLarge /= float64(numRuns)
	avgTotalColdStartLarge /= float64(numRuns)
	avgStartRSSLarge /= float64(numRuns)
	avgPeakRSSLarge /= float64(numRuns)

	// 计算轻模型平均值
	var avgSessionCreationSmall, avgModelLoadingSmall, avgFirstInferenceSmall, avgTotalColdStartSmall float64
	var avgStartRSSSmall, avgPeakRSSSmall float64

	for _, r := range resultsSmall {
		avgSessionCreationSmall += r.SessionCreationTime
		avgModelLoadingSmall += r.ModelLoadingTime
		avgFirstInferenceSmall += r.FirstInferenceTime
		avgTotalColdStartSmall += r.TotalColdStartTime
		avgStartRSSSmall += r.StartRSS
		avgPeakRSSSmall += r.PeakRSS
	}

	avgSessionCreationSmall /= float64(numRuns)
	avgModelLoadingSmall /= float64(numRuns)
	avgFirstInferenceSmall /= float64(numRuns)
	avgTotalColdStartSmall /= float64(numRuns)
	avgStartRSSSmall /= float64(numRuns)
	avgPeakRSSSmall /= float64(numRuns)

	fmt.Println("\n===== 大模型 (YOLO11x) 20次测试平均值 =====")
	// 中间数据保留5位小数
	fmt.Printf("会话创建时间: %.5f ms\n", avgSessionCreationLarge)
	fmt.Printf("模型加载时间: %.5f ms\n", avgModelLoadingLarge)
	fmt.Printf("首次推理时间: %.5f ms\n", avgFirstInferenceLarge)
	fmt.Printf("总冷启动时间: %.5f ms\n", avgTotalColdStartLarge)
	fmt.Printf("Start RSS: %.5f MB\n", avgStartRSSLarge)
	fmt.Printf("Peak RSS: %.5f MB\n\n", avgPeakRSSLarge)

	fmt.Println("===== 轻模型 (YOLO11n) 20次测试平均值 =====")
	// 中间数据保留5位小数
	fmt.Printf("会话创建时间: %.5f ms\n", avgSessionCreationSmall)
	fmt.Printf("模型加载时间: %.5f ms\n", avgModelLoadingSmall)
	fmt.Printf("首次推理时间: %.5f ms\n", avgFirstInferenceSmall)
	fmt.Printf("总冷启动时间: %.5f ms\n", avgTotalColdStartSmall)
	fmt.Printf("Start RSS: %.5f MB\n", avgStartRSSSmall)
	fmt.Printf("Peak RSS: %.5f MB\n", avgPeakRSSSmall)

	// 保存结果
	resultsDir := filepath.Join(basePath, "results")
	os.MkdirAll(resultsDir, 0755)

	resultPath := filepath.Join(resultsDir, "go_cold_start_decomposition_result.txt")
	resultFile, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer resultFile.Close()

	fmt.Fprintf(resultFile, "===== Go 冷启动分解测试结果 =====\n\n")

	fmt.Fprintf(resultFile, "===== 大模型 (YOLO11x) =====\n")
	for i, r := range resultsLarge {
		fmt.Fprintf(resultFile, "===== 第 %d 次测试 =====\n", i+1)
		fmt.Fprintf(resultFile, "会话创建时间: %.5f ms\n", r.SessionCreationTime)
		fmt.Fprintf(resultFile, "模型加载时间: %.5f ms\n", r.ModelLoadingTime)
		fmt.Fprintf(resultFile, "首次推理时间: %.5f ms\n", r.FirstInferenceTime)
		fmt.Fprintf(resultFile, "总冷启动时间: %.5f ms\n", r.TotalColdStartTime)
		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Fprintf(resultFile, "Start RSS: %.5f MB\n", r.StartRSS)
		fmt.Fprintf(resultFile, "Peak RSS: %.5f MB\n\n", r.PeakRSS)
	}

	fmt.Fprintf(resultFile, "===== 大模型 (YOLO11x) 20次测试平均值 =====\n")
	fmt.Fprintf(resultFile, "会话创建时间: %.5f ms\n", avgSessionCreationLarge)
	fmt.Fprintf(resultFile, "模型加载时间: %.5f ms\n", avgModelLoadingLarge)
	fmt.Fprintf(resultFile, "首次推理时间: %.5f ms\n", avgFirstInferenceLarge)
	fmt.Fprintf(resultFile, "总冷启动时间: %.5f ms\n", avgTotalColdStartLarge)
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(resultFile, "Start RSS: %.5f MB\n", avgStartRSSLarge)
	fmt.Fprintf(resultFile, "Peak RSS: %.5f MB\n\n", avgPeakRSSLarge)

	fmt.Fprintf(resultFile, "===== 轻模型 (YOLO11n) =====\n")
	for i, r := range resultsSmall {
		fmt.Fprintf(resultFile, "===== 第 %d 次测试 =====\n", i+1)
		fmt.Fprintf(resultFile, "会话创建时间: %.5f ms\n", r.SessionCreationTime)
		fmt.Fprintf(resultFile, "模型加载时间: %.5f ms\n", r.ModelLoadingTime)
		fmt.Fprintf(resultFile, "首次推理时间: %.5f ms\n", r.FirstInferenceTime)
		fmt.Fprintf(resultFile, "总冷启动时间: %.5f ms\n", r.TotalColdStartTime)
		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Fprintf(resultFile, "Start RSS: %.5f MB\n", r.StartRSS)
		fmt.Fprintf(resultFile, "Peak RSS: %.5f MB\n\n", r.PeakRSS)
	}

	fmt.Fprintf(resultFile, "===== 轻模型 (YOLO11n) 20次测试平均值 =====\n")
	fmt.Fprintf(resultFile, "会话创建时间: %.5f ms\n", avgSessionCreationSmall)
	fmt.Fprintf(resultFile, "模型加载时间: %.5f ms\n", avgModelLoadingSmall)
	fmt.Fprintf(resultFile, "首次推理时间: %.5f ms\n", avgFirstInferenceSmall)
	fmt.Fprintf(resultFile, "总冷启动时间: %.5f ms\n", avgTotalColdStartSmall)
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(resultFile, "Start RSS: %.5f MB\n", avgStartRSSSmall)
	fmt.Fprintf(resultFile, "Peak RSS: %.5f MB\n", avgPeakRSSSmall)

	fmt.Printf("\n结果已保存到: %s\n", resultPath)
	fmt.Println("测试完成!")
}
