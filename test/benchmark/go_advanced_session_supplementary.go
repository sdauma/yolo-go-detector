// go_advanced_session_supplementary.go
// Go AdvancedSession 补充实验（工程级接口能力评估）
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置动态线程数
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 非语言级性能比较，仅评估 AdvancedSession 接口的工程能力
//
// 测试目的：
// - 验证 NewAdvancedSession 的 I/O Binding 能力
// - 统计 Tensor 分配计数、Session 创建计数等工程指标
// - 作为 NewSession vs AdvancedSession 接口差异的补充说明

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

type PerformanceMetrics struct {
	Avg float64
	P50 float64
	P90 float64
	P99 float64
	Min float64
	Max float64
}

type EngineeringMetrics struct {
	TensorAllocationCount int
	IOBindingEnabled      bool
	SessionCreationCount  int
	PeakRSS               float64
}

func main() {
	fmt.Println("===== Go AdvancedSession 补充实验 =====")
	fmt.Println("实验性质：工程级接口能力评估（非语言级性能比较）")
	fmt.Println()

	currentDir, _ := os.Getwd()
	projectRoot := findProjectRoot(currentDir)

	modelPath := filepath.Join(projectRoot, "third_party", "yolo11x.onnx")
	libraryPath := filepath.Join(projectRoot, "third_party", "onnxruntime.dll")

	fmt.Printf("当前目录: %s\n", currentDir)
	fmt.Printf("项目根路径: %s\n", projectRoot)
	fmt.Printf("模型路径: %s\n", modelPath)
	fmt.Printf("库路径: %s\n", libraryPath)

	ort.SetSharedLibraryPath(libraryPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("ONNX Runtime 环境初始化失败: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	fmt.Printf("当前目录: %s\n", currentDir)
	fmt.Printf("项目根路径: %s\n", projectRoot)
	fmt.Printf("模型路径: %s\n", modelPath)
	fmt.Printf("库路径: %s\n", libraryPath)
	fmt.Println("ONNX Runtime 环境初始化成功!")
	fmt.Println()

	threadConfigs := []int{1, 2, 4, 8}
	results := make(map[int]PerformanceMetrics)
	engineeringResults := make(map[int]EngineeringMetrics)

	for i, numThreads := range threadConfigs {
		fmt.Printf("===== 实验编号 S-A%d: intra_op_num_threads=%d =====\n", i+1, numThreads)

		perfMetrics, engMetrics := runAdvancedSessionTest(modelPath, numThreads, projectRoot)
		results[numThreads] = perfMetrics
		engineeringResults[numThreads] = engMetrics

		fmt.Printf("性能指标: avg=%.5f ms, p50=%.5f ms, p90=%.5f ms, p99=%.5f ms, min=%.5f ms, max=%.5f ms\n",
			perfMetrics.Avg, perfMetrics.P50, perfMetrics.P90, perfMetrics.P99, perfMetrics.Min, perfMetrics.Max)
		fmt.Printf("工程指标: Tensor分配次数=%d, I/O Binding=%t, Session创建次数=%d, 峰值RSS=%.5f MB\n",
			engMetrics.TensorAllocationCount, engMetrics.IOBindingEnabled, engMetrics.SessionCreationCount, engMetrics.PeakRSS)
		fmt.Println()
	}

	saveResults(results, engineeringResults)
	fmt.Println("===== 补充实验完成 =====")
}

func runAdvancedSessionTest(modelPath string, numThreads int, projectRoot string) (PerformanceMetrics, EngineeringMetrics) {
	engMetrics := EngineeringMetrics{
		TensorAllocationCount: 0,
		IOBindingEnabled:      true,
		SessionCreationCount:  1,
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("创建SessionOptions失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(numThreads)
	opts.SetInterOpNumThreads(1)

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入张量失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}
	defer inputTensor.Destroy()

	// 从预生成的二进制文件加载输入数据
	inputDataPath := filepath.Join(projectRoot, "test", "data", "input_data.bin")
	inputDataFile, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据文件失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}

	// 转换为 float32 并填充到张量
	inputData := inputTensor.GetData()
	expectedSize := 1 * 3 * 640 * 640 * 4 // float32 = 4 bytes
	if len(inputDataFile) != expectedSize {
		fmt.Printf("输入数据文件大小不匹配: 期望 %d 字节，实际 %d 字节\n", expectedSize, len(inputDataFile))
		return PerformanceMetrics{}, engMetrics
	}

	// 将字节数据转换为 float32 并复制到张量
	for i := 0; i < len(inputData); i += 4 {
		bits := binary.LittleEndian.Uint32(inputDataFile[i : i+4])
		value := math.Float32frombits(bits)
		inputData[i/4] = value
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("创建输出张量失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}
	defer outputTensor.Destroy()

	engMetrics.TensorAllocationCount = 2

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, opts)
	if err != nil {
		fmt.Printf("创建AdvancedSession失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}
	defer session.Destroy()

	fmt.Println("AdvancedSession 创建成功!")
	fmt.Printf("线程配置: intra_op_num_threads=%d, inter_op_num_threads=1\n", numThreads)

	startRSS := getProcessRSS()
	fmt.Printf("Start RSS: %.5f MB\n", startRSS)

	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		err := session.Run()
		if err != nil {
			fmt.Printf("Warmup 运行失败: %v\n", err)
			return PerformanceMetrics{}, engMetrics
		}
	}

	warmupRSS := getProcessRSS()
	fmt.Printf("Warmup 后 RSS: %.5f MB\n", warmupRSS)

	fmt.Println("开始基准测试...")
	latencies := make([]float64, 100)
	for i := 0; i < 100; i++ {
		start := time.Now()
		err := session.Run()
		if err != nil {
			fmt.Printf("运行失败: %v\n", err)
			return PerformanceMetrics{}, engMetrics
		}
		elapsed := time.Since(start).Seconds() * 1000.0
		latencies[i] = elapsed
		if i < 5 {
			fmt.Printf("  推理 %d: %.5f ms\n", i+1, elapsed)
		}
	}

	engMetrics.PeakRSS = getProcessRSS()
	fmt.Printf("Peak RSS: %.5f MB\n", engMetrics.PeakRSS)

	return calculateMetrics(latencies), engMetrics
}

func calculateMetrics(latencies []float64) PerformanceMetrics {
	if len(latencies) == 0 {
		return PerformanceMetrics{}
	}

	sum := 0.0
	min := latencies[0]
	max := latencies[0]
	for _, lat := range latencies {
		sum += lat
		if lat < min {
			min = lat
		}
		if lat > max {
			max = lat
		}
	}
	avg := sum / float64(len(latencies))

	sorted := make([]float64, len(latencies))
	copy(sorted, latencies)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	p50 := sorted[int(float64(len(sorted))*0.5)]
	p90 := sorted[int(float64(len(sorted))*0.9)]
	p99 := sorted[int(float64(len(sorted))*0.99)]

	return PerformanceMetrics{
		Avg: avg,
		P50: p50,
		P90: p90,
		P99: p99,
		Min: min,
		Max: max,
	}
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

func findProjectRoot(currentDir string) string {
	fmt.Printf("调试: 开始查找项目根目录，当前目录: %s\n", currentDir)
	for {
		testPath := filepath.Join(currentDir, "third_party", "yolo11x.onnx")
		fmt.Printf("调试: 检查路径: %s\n", testPath)
		if _, err := os.Stat(testPath); err == nil {
			fmt.Printf("调试: 找到项目根目录: %s\n", currentDir)
			return currentDir
		}

		currentDir = filepath.Dir(currentDir)
		if currentDir == "." || currentDir == "/" {
			fmt.Printf("调试: 已到达根目录，返回: %s\n", currentDir)
			return currentDir
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func saveResults(results map[int]PerformanceMetrics, engineeringResults map[int]EngineeringMetrics) {
	currentDir, _ := os.Getwd()
	projectRoot := findProjectRoot(currentDir)
	resultPath := filepath.Join(projectRoot, "results", "go_advanced_session_supplementary.txt")

	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer file.Close()

	file.WriteString("===== Go AdvancedSession 补充实验结果 =====\n")
	file.WriteString("实验性质：工程级接口能力评估（非语言级性能比较）\n")
	file.WriteString("执行路径：AdvancedSession + I/O Binding + 预分配 Tensor\n")
	file.WriteString("对照策略：Python 仍使用 baseline（不启用 io_binding）\n\n")

	file.WriteString("性能指标：\n")
	file.WriteString("线程配置\t平均延迟\tP50\tP90\tP99\t最小值\t最大值\n")
	for _, numThreads := range []int{1, 2, 4, 8} {
		metrics := results[numThreads]
		file.WriteString(fmt.Sprintf("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n",
			numThreads, metrics.Avg, metrics.P50, metrics.P90, metrics.P99, metrics.Min, metrics.Max))
	}

	file.WriteString("\n工程指标：\n")
	file.WriteString("线程配置\tTensor分配次数\tI/O Binding\tSession创建次数\t峰值RSS(MB)\n")
	for _, numThreads := range []int{1, 2, 4, 8} {
		metrics := engineeringResults[numThreads]
		file.WriteString(fmt.Sprintf("%d\t%d\t%t\t%d\t%.5f\n",
			numThreads, metrics.TensorAllocationCount, metrics.IOBindingEnabled,
			metrics.SessionCreationCount, metrics.PeakRSS))
	}

	file.WriteString("\n不可比声明：\n")
	file.WriteString("本节实验通过 AdvancedSession 与 I/O Binding 引入了工程级执行路径优化，\n")
	file.WriteString("其内存分配和执行调度机制与前文 baseline 测试存在本质差异，\n")
	file.WriteString("因此结果不用于修正语言级性能结论，仅用于评估 Go 在 ONNX 推理任务中的工程接口性能潜力。\n")

	fmt.Printf("结果已保存到: %s\n", resultPath)
}
