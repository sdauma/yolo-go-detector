package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
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

	modelPath := projectRoot + "\\third_party\\yolo11x.onnx"
	libraryPath := projectRoot + "\\third_party\\onnxruntime.dll"

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

		perfMetrics, engMetrics := runAdvancedSessionTest(modelPath, numThreads)
		results[numThreads] = perfMetrics
		engineeringResults[numThreads] = engMetrics

		fmt.Printf("性能指标: avg=%.2f ms, p50=%.2f ms, p90=%.2f ms, p99=%.2f ms, min=%.2f ms, max=%.2f ms\n",
			perfMetrics.Avg, perfMetrics.P50, perfMetrics.P90, perfMetrics.P99, perfMetrics.Min, perfMetrics.Max)
		fmt.Printf("工程指标: Tensor分配次数=%d, I/O Binding=%t, Session创建次数=%d, 峰值RSS=%.2f MB\n",
			engMetrics.TensorAllocationCount, engMetrics.IOBindingEnabled, engMetrics.SessionCreationCount, engMetrics.PeakRSS)
		fmt.Println()
	}

	saveResults(results, engineeringResults)
	fmt.Println("===== 补充实验完成 =====")
}

func runAdvancedSessionTest(modelPath string, numThreads int) (PerformanceMetrics, EngineeringMetrics) {
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
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, opts)
	if err != nil {
		fmt.Printf("创建AdvancedSession失败: %v\n", err)
		return PerformanceMetrics{}, engMetrics
	}
	defer session.Destroy()

	fmt.Println("AdvancedSession 创建成功!")
	fmt.Printf("线程配置: intra_op_num_threads=%d, inter_op_num_threads=1\n", numThreads)

	startRSS := getProcessRSS()
	fmt.Printf("Start RSS: %.2f MB\n", startRSS)

	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		err := session.Run()
		if err != nil {
			fmt.Printf("Warmup 运行失败: %v\n", err)
			return PerformanceMetrics{}, engMetrics
		}
	}

	warmupRSS := getProcessRSS()
	fmt.Printf("Warmup 后 RSS: %.2f MB\n", warmupRSS)

	fmt.Println("开始基准测试...")
	latencies := make([]float64, 100)
	for i := 0; i < 100; i++ {
		start := time.Now()
		err := session.Run()
		if err != nil {
			fmt.Printf("运行失败: %v\n", err)
			return PerformanceMetrics{}, engMetrics
		}
		elapsed := time.Since(start).Milliseconds()
		latencies[i] = float64(elapsed)
	}

	engMetrics.PeakRSS = getProcessRSS()
	fmt.Printf("Peak RSS: %.2f MB\n", engMetrics.PeakRSS)

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

	p50 := sorted[len(sorted)*50/100]
	p90 := sorted[len(sorted)*90/100]
	p99 := sorted[len(sorted)*99/100]

	return PerformanceMetrics{
		Avg: avg,
		P50: p50,
		P90: p90,
		P99: p99,
		Min: min,
		Max: max,
	}
}

func getProcessRSS() float64 {
	cmd := exec.Command("powershell", "-Command", "(Get-Process -Id $PID).WorkingSet64 / 1MB")
	cmd.Env = append(os.Environ(), fmt.Sprintf("PID=%d", os.Getpid()))
	output, err := cmd.Output()
	if err != nil {
		return 0
	}
	rssStr := strings.TrimSpace(string(output))
	rss, err := strconv.ParseFloat(rssStr, 64)
	if err != nil {
		return 0
	}
	return rss
}

func findProjectRoot(currentDir string) string {
	fmt.Printf("调试: 开始查找项目根目录，当前目录: %s\n", currentDir)
	for {
		testPath := currentDir + "\\third_party\\yolo11x.onnx"
		fmt.Printf("调试: 检查路径: %s\n", testPath)
		if _, err := os.Stat(testPath); err == nil {
			fmt.Printf("调试: 找到项目根目录: %s\n", currentDir)
			return currentDir
		}

		lastSlashIndex := lastIndexOf(currentDir, "\\")
		if lastSlashIndex <= 0 {
			fmt.Printf("调试: 已到达根目录，返回: %s\n", currentDir)
			return currentDir
		}
		currentDir = currentDir[:lastSlashIndex]
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func lastIndexOf(s, substr string) int {
	for i := len(s) - len(substr); i >= 0; i-- {
		if len(s) >= i+len(substr) && s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func saveResults(results map[int]PerformanceMetrics, engineeringResults map[int]EngineeringMetrics) {
	currentDir, _ := os.Getwd()
	projectRoot := findProjectRoot(currentDir)
	resultPath := projectRoot + "\\results\\go_advanced_session_supplementary.txt"

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
		file.WriteString(fmt.Sprintf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
			numThreads, metrics.Avg, metrics.P50, metrics.P90, metrics.P99, metrics.Min, metrics.Max))
	}

	file.WriteString("\n工程指标：\n")
	file.WriteString("线程配置\tTensor分配次数\tI/O Binding\tSession创建次数\t峰值RSS(MB)\n")
	for _, numThreads := range []int{1, 2, 4, 8} {
		metrics := engineeringResults[numThreads]
		file.WriteString(fmt.Sprintf("%d\t%d\t%t\t%d\t%.2f\n",
			numThreads, metrics.TensorAllocationCount, metrics.IOBindingEnabled,
			metrics.SessionCreationCount, metrics.PeakRSS))
	}

	file.WriteString("\n不可比声明：\n")
	file.WriteString("本节实验通过 AdvancedSession 与 I/O Binding 引入了工程级执行路径优化，\n")
	file.WriteString("其内存分配和执行调度机制与前文 baseline 测试存在本质差异，\n")
	file.WriteString("因此结果不用于修正语言级性能结论，仅用于评估 Go 在 ONNX 推理任务中的工程接口性能潜力。\n")

	fmt.Printf("结果已保存到: %s\n", resultPath)
}
