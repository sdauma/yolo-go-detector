// go_memory_standardization.go
// Go 内存标准化测试
//
// 技术说明：
// - 使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
//   自动启用 I/O Binding，但不接受 SessionOptions 参数
// - 线程配置由 ONNX Runtime 默认行为决定（intra_op_num_threads 默认等于 CPU 核数）
// - 代码中创建了 SessionOptions 并设置了 intraOp=12，但由于 NewSession 不接受 opts，
//   这些设置实际上不生效。保留 opts 创建代码仅用于记录意图
//
// 测试目的：
// - 记录解释器常驻内存（基础内存）
// - 记录模型加载后的内存
// - 记录推理后的内存
// - 执行多次测试，计算平均值
// - 确保数据稳定性和可重复性

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

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

// MemoryResult 内存测试结果
type MemoryResult struct {
	InterpreterMemory   float64 // 解释器常驻内存
	ModelLoadedMemory   float64 // 模型加载后内存
	PostInferenceMemory float64 // 推理后内存
	MemoryIncrease      float64 // 内存增加量（模型加载+推理）
	GoHeapMemory        float64 // Go堆内存
}

// runMemoryTest 执行一次内存测试
func runMemoryTest(modelPath, modelName, libPath string) (*MemoryResult, error) {
	fmt.Printf("\n===== Go 内存测试 - %s ====\n", modelName)

	// 1. 测量解释器常驻内存（基础内存）
	fmt.Println("测量解释器常驻内存...")
	interpreterMemory := getProcessRSS()

	// 获取Go堆内存
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	goHeapMemory := float64(m.Alloc) / 1024 / 1024

	fmt.Printf("解释器常驻内存: %.2f MB\n", interpreterMemory)
	fmt.Printf("Go堆内存: %.2f MB\n", goHeapMemory)

	// 2. 初始化ORT环境并加载模型
	fmt.Println("加载模型...")
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 创建会话选项
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

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %v", err)
	}
	defer inputTensor.Destroy()

	// 从预生成的二进制文件加载输入数据
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

	// 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输出张量失败: %v", err)
	}
	defer outputTensor.Destroy()

	// 创建会话（加载模型）
	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("创建会话失败: %v", err)
	}
	defer session.Destroy()

	// 3. 测量模型加载后的内存
	modelLoadedMemory := getProcessRSS()
	fmt.Printf("模型加载后内存: %.2f MB\n", modelLoadedMemory)

	// 4. 执行推理
	fmt.Println("执行推理...")
	for i := 0; i < 10; i++ { // 执行10次推理以稳定内存使用
		if err := session.Run(); err != nil {
			return nil, fmt.Errorf("推理失败: %v", err)
		}
	}

	// 5. 测量推理后的内存
	postInferenceMemory := getProcessRSS()
	fmt.Printf("推理后内存: %.2f MB\n", postInferenceMemory)

	// 计算内存增加量
	memoryIncrease := postInferenceMemory - interpreterMemory
	fmt.Printf("内存增加量: %.2f MB\n", memoryIncrease)

	return &MemoryResult{
		InterpreterMemory:   interpreterMemory,
		ModelLoadedMemory:   modelLoadedMemory,
		PostInferenceMemory: postInferenceMemory,
		MemoryIncrease:      memoryIncrease,
		GoHeapMemory:        goHeapMemory,
	}, nil
}

func main() {
	fmt.Println("===== Go 内存标准化测试（10次运行）=====")

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

	// 运行10次测试 - 大模型
	numRuns := 10
	resultsLarge := make([]*MemoryResult, numRuns)
	resultsSmall := make([]*MemoryResult, numRuns)

	fmt.Println("\n===== 测试大模型 (YOLO11x) =====")
	for i := 0; i < numRuns; i++ {
		fmt.Printf("\n===== 第 %d 次测试 =====\n", i+1)
		result, err := runMemoryTest(modelPathLarge, "YOLO11x", libPath)
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		resultsLarge[i] = result
	}

	fmt.Println("\n===== 测试轻模型 (YOLO11n) =====")
	for i := 0; i < numRuns; i++ {
		fmt.Printf("\n===== 第 %d 次测试 =====\n", i+1)
		result, err := runMemoryTest(modelPathSmall, "YOLO11n", libPath)
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		resultsSmall[i] = result
	}

	// 计算大模型平均值
	var avgInterpreterLarge, avgModelLoadedLarge, avgPostInferenceLarge, avgMemoryIncreaseLarge, avgGoHeapLarge float64

	for _, r := range resultsLarge {
		avgInterpreterLarge += r.InterpreterMemory
		avgModelLoadedLarge += r.ModelLoadedMemory
		avgPostInferenceLarge += r.PostInferenceMemory
		avgMemoryIncreaseLarge += r.MemoryIncrease
		avgGoHeapLarge += r.GoHeapMemory
	}

	avgInterpreterLarge /= float64(numRuns)
	avgModelLoadedLarge /= float64(numRuns)
	avgPostInferenceLarge /= float64(numRuns)
	avgMemoryIncreaseLarge /= float64(numRuns)
	avgGoHeapLarge /= float64(numRuns)

	// 计算轻模型平均值
	var avgInterpreterSmall, avgModelLoadedSmall, avgPostInferenceSmall, avgMemoryIncreaseSmall, avgGoHeapSmall float64

	for _, r := range resultsSmall {
		avgInterpreterSmall += r.InterpreterMemory
		avgModelLoadedSmall += r.ModelLoadedMemory
		avgPostInferenceSmall += r.PostInferenceMemory
		avgMemoryIncreaseSmall += r.MemoryIncrease
		avgGoHeapSmall += r.GoHeapMemory
	}

	avgInterpreterSmall /= float64(numRuns)
	avgModelLoadedSmall /= float64(numRuns)
	avgPostInferenceSmall /= float64(numRuns)
	avgMemoryIncreaseSmall /= float64(numRuns)
	avgGoHeapSmall /= float64(numRuns)

	fmt.Println("\n===== 大模型 (YOLO11x) 10次测试平均值 =====")
	fmt.Printf("解释器常驻内存: %.2f MB\n", avgInterpreterLarge)
	fmt.Printf("模型加载后内存: %.2f MB\n", avgModelLoadedLarge)
	fmt.Printf("推理后内存: %.2f MB\n", avgPostInferenceLarge)
	fmt.Printf("内存增加量: %.2f MB\n", avgMemoryIncreaseLarge)
	fmt.Printf("Go堆内存: %.2f MB\n", avgGoHeapLarge)

	fmt.Println("\n===== 轻模型 (YOLO11n) 10次测试平均值 =====")
	fmt.Printf("解释器常驻内存: %.2f MB\n", avgInterpreterSmall)
	fmt.Printf("模型加载后内存: %.2f MB\n", avgModelLoadedSmall)
	fmt.Printf("推理后内存: %.2f MB\n", avgPostInferenceSmall)
	fmt.Printf("内存增加量: %.2f MB\n", avgMemoryIncreaseSmall)
	fmt.Printf("Go堆内存: %.2f MB\n", avgGoHeapSmall)

	// 保存结果
	resultsDir := filepath.Join(basePath, "results")
	os.MkdirAll(resultsDir, 0755)

	resultPath := filepath.Join(resultsDir, "go_memory_standardization_result.txt")
	resultFile, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer resultFile.Close()

	fmt.Fprintf(resultFile, "===== Go 内存标准化测试结果 =====\n\n")

	fmt.Fprintf(resultFile, "===== 大模型 (YOLO11x) =====\n")
	for i, r := range resultsLarge {
		fmt.Fprintf(resultFile, "===== 第 %d 次测试 =====\n", i+1)
		fmt.Fprintf(resultFile, "解释器常驻内存: %.5f MB\n", r.InterpreterMemory)
		fmt.Fprintf(resultFile, "模型加载后内存: %.5f MB\n", r.ModelLoadedMemory)
		fmt.Fprintf(resultFile, "推理后内存: %.5f MB\n", r.PostInferenceMemory)
		fmt.Fprintf(resultFile, "内存增加量: %.5f MB\n", r.MemoryIncrease)
		fmt.Fprintf(resultFile, "Go堆内存: %.5f MB\n\n", r.GoHeapMemory)
	}

	fmt.Fprintf(resultFile, "===== 大模型 (YOLO11x) 10次测试平均值 =====\n")
	fmt.Fprintf(resultFile, "解释器常驻内存: %.5f MB\n", avgInterpreterLarge)
	fmt.Fprintf(resultFile, "模型加载后内存: %.5f MB\n", avgModelLoadedLarge)
	fmt.Fprintf(resultFile, "推理后内存: %.5f MB\n", avgPostInferenceLarge)
	fmt.Fprintf(resultFile, "内存增加量: %.5f MB\n", avgMemoryIncreaseLarge)
	fmt.Fprintf(resultFile, "Go堆内存: %.5f MB\n\n", avgGoHeapLarge)

	fmt.Fprintf(resultFile, "===== 轻模型 (YOLO11n) =====\n")
	for i, r := range resultsSmall {
		fmt.Fprintf(resultFile, "===== 第 %d 次测试 =====\n", i+1)
		fmt.Fprintf(resultFile, "解释器常驻内存: %.5f MB\n", r.InterpreterMemory)
		fmt.Fprintf(resultFile, "模型加载后内存: %.5f MB\n", r.ModelLoadedMemory)
		fmt.Fprintf(resultFile, "推理后内存: %.5f MB\n", r.PostInferenceMemory)
		fmt.Fprintf(resultFile, "内存增加量: %.5f MB\n", r.MemoryIncrease)
		fmt.Fprintf(resultFile, "Go堆内存: %.5f MB\n\n", r.GoHeapMemory)
	}

	fmt.Fprintf(resultFile, "===== 轻模型 (YOLO11n) 10次测试平均值 =====\n")
	fmt.Fprintf(resultFile, "解释器常驻内存: %.5f MB\n", avgInterpreterSmall)
	fmt.Fprintf(resultFile, "模型加载后内存: %.5f MB\n", avgModelLoadedSmall)
	fmt.Fprintf(resultFile, "推理后内存: %.5f MB\n", avgPostInferenceSmall)
	fmt.Fprintf(resultFile, "内存增加量: %.5f MB\n", avgMemoryIncreaseSmall)
	fmt.Fprintf(resultFile, "Go堆内存: %.5f MB\n", avgGoHeapSmall)

	fmt.Printf("\n结果已保存到: %s\n", resultPath)
	fmt.Println("测试完成!")
}
