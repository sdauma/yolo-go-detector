// go_concurrent_architecture_comparison.go
// Go 并发架构对比测试（学术版：Shared / Mutex / Session Pool）
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=1, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 测试并发度 1/2/4/6/8/12，每并发度 500 次请求
// - Shared Session 中每个 goroutine 创建独立 Tensor；Session Pool 中每个 goroutine 创建独立 Session
//
// 测试目的：
// - 在更大并发范围内对比三种架构的吞吐量、延迟、内存表现
// - 为论文提供详尽的架构对比数据

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

type ArchitectureTestResult struct {
	Architecture       string
	Concurrency        int
	TotalRequests      int
	SuccessfulRequests int
	FailedRequests     int
	TotalTime          float64
	AvgLatency         float64
	P50Latency         float64
	P90Latency         float64
	P99Latency         float64
	MinLatency         float64
	MaxLatency         float64
	Throughput         float64
	StartRSS           float64
	PeakRSS            float64
	EndRSS             float64
	RSSDrift           float64
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

func calculatePercentile(sorted []float64, percentile float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(float64(len(sorted)) * percentile)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

// ========================================
// 架构1: Shared Session - 多线程共享同一个 Session
//
// 注意：本测试有意让多个 goroutine 共享同一个 Session 并行调用 Run()，
// 目的是观察 ONNX Runtime 内部资源竞争对吞吐量和延迟的影响。
// peakRSS 的并发读取（无锁）是已知的低影响数据竞争，仅用于记录内存峰值趋势。
// ========================================
func runSharedSessionTest(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
) *ArchitectureTestResult {
	fmt.Printf("  [Shared Session] 测试: %d 并发, %d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	// 创建共享 Session
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("  创建 SessionOptions 失败: %v\n", err)
		return nil
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("  创建输入 Tensor 失败: %v\n", err)
		return nil
	}
	defer inputTensor.Destroy()

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("  创建输出 Tensor 失败: %v\n", err)
		return nil
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		opts,
	)
	if err != nil {
		fmt.Printf("  创建 Session 失败: %v\n", err)
		return nil
	}
	defer session.Destroy()

	// 并发测试
	var wg sync.WaitGroup
	latencies := make(chan float64, numRequests)
	errors := make(chan error, numRequests)

	startTime := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			batchSize := numRequests / concurrency
			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				start := time.Now()
				err := session.Run()
				latency := float64(time.Since(start).Milliseconds())

				if err != nil {
					errors <- err
				} else {
					latencies <- latency
				}
			}
		}()
	}

	wg.Wait()
	close(latencies)
	close(errors)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	latencyList := make([]float64, 0, numRequests)
	for lat := range latencies {
		latencyList = append(latencyList, lat)
	}

	errorCount := 0
	for range errors {
		errorCount++
	}

	if len(latencyList) == 0 {
		return &ArchitectureTestResult{
			Architecture:   "Shared Session",
			Concurrency:    concurrency,
			TotalRequests:  numRequests,
			FailedRequests: numRequests,
			TotalTime:      totalTime,
		}
	}

	sort.Float64s(latencyList)

	sumLatency := 0.0
	for _, lat := range latencyList {
		sumLatency += lat
	}

	return &ArchitectureTestResult{
		Architecture:       "Shared Session",
		Concurrency:        concurrency,
		TotalRequests:      numRequests,
		SuccessfulRequests: len(latencyList),
		FailedRequests:     errorCount,
		TotalTime:          totalTime,
		AvgLatency:         sumLatency / float64(len(latencyList)),
		P50Latency:         calculatePercentile(latencyList, 0.50),
		P90Latency:         calculatePercentile(latencyList, 0.90),
		P99Latency:         calculatePercentile(latencyList, 0.99),
		MinLatency:         latencyList[0],
		MaxLatency:         latencyList[len(latencyList)-1],
		Throughput:         float64(len(latencyList)) / (totalTime / 1000.0),
		StartRSS:           startRSS,
		PeakRSS:            peakRSS,
		EndRSS:             endRSS,
		RSSDrift:           endRSS - startRSS,
	}
}

// ========================================
// 架构2: Mutex - 串行化访问共享 Session
// ========================================
func runMutexTest(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
) *ArchitectureTestResult {
	fmt.Printf("  [Mutex] 测试: %d 并发, %d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	// 创建共享 Session
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("  创建 SessionOptions 失败: %v\n", err)
		return nil
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("  创建输入 Tensor 失败: %v\n", err)
		return nil
	}
	defer inputTensor.Destroy()

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("  创建输出 Tensor 失败: %v\n", err)
		return nil
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		opts,
	)
	if err != nil {
		fmt.Printf("  创建 Session 失败: %v\n", err)
		return nil
	}
	defer session.Destroy()

	// 使用 Mutex 串行化
	var mutex sync.Mutex

	var wg sync.WaitGroup
	latencies := make(chan float64, numRequests)
	errors := make(chan error, numRequests)

	startTime := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			batchSize := numRequests / concurrency
			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				mutex.Lock()
				start := time.Now()
				err := session.Run()
				latency := float64(time.Since(start).Milliseconds())
				mutex.Unlock()

				if err != nil {
					errors <- err
				} else {
					latencies <- latency
				}
			}
		}()
	}

	wg.Wait()
	close(latencies)
	close(errors)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	latencyList := make([]float64, 0, numRequests)
	for lat := range latencies {
		latencyList = append(latencyList, lat)
	}

	errorCount := 0
	for range errors {
		errorCount++
	}

	if len(latencyList) == 0 {
		return &ArchitectureTestResult{
			Architecture:   "Mutex",
			Concurrency:    concurrency,
			TotalRequests:  numRequests,
			FailedRequests: numRequests,
			TotalTime:      totalTime,
		}
	}

	sort.Float64s(latencyList)

	sumLatency := 0.0
	for _, lat := range latencyList {
		sumLatency += lat
	}

	return &ArchitectureTestResult{
		Architecture:       "Mutex",
		Concurrency:        concurrency,
		TotalRequests:      numRequests,
		SuccessfulRequests: len(latencyList),
		FailedRequests:     errorCount,
		TotalTime:          totalTime,
		AvgLatency:         sumLatency / float64(len(latencyList)),
		P50Latency:         calculatePercentile(latencyList, 0.50),
		P90Latency:         calculatePercentile(latencyList, 0.90),
		P99Latency:         calculatePercentile(latencyList, 0.99),
		MinLatency:         latencyList[0],
		MaxLatency:         latencyList[len(latencyList)-1],
		Throughput:         float64(len(latencyList)) / (totalTime / 1000.0),
		StartRSS:           startRSS,
		PeakRSS:            peakRSS,
		EndRSS:             endRSS,
		RSSDrift:           endRSS - startRSS,
	}
}

// ========================================
// 架构3: Session Pool - 每个 goroutine 独立 Session
// ========================================
func runSessionPoolTest(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
) *ArchitectureTestResult {
	fmt.Printf("  [Session Pool] 测试: %d 并发, %d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	var wg sync.WaitGroup
	latencies := make(chan float64, numRequests)
	errors := make(chan error, numRequests)

	startTime := time.Now()

	// 每个 goroutine 创建独立 Session
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// 创建独立 Session
			opts, err := ort.NewSessionOptions()
			if err != nil {
				fmt.Printf("  创建 SessionOptions 失败: %v\n", err)
				return
			}
			defer opts.Destroy()

			opts.SetIntraOpNumThreads(1)
			opts.SetInterOpNumThreads(1)

			inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
			if err != nil {
				fmt.Printf("  创建输入 Tensor 失败: %v\n", err)
				return
			}
			defer inputTensor.Destroy()

			floatData := inputTensor.GetData()
			for j := 0; j < len(floatData); j++ {
				if j*4 < len(inputData) {
					bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
					floatData[j] = math.Float32frombits(bits)
				}
			}

			outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
			if err != nil {
				fmt.Printf("  创建输出 Tensor 失败: %v\n", err)
				return
			}
			defer outputTensor.Destroy()

			session, err := ort.NewAdvancedSession(
				modelPath,
				[]string{"images"},
				[]string{"output0"},
				[]ort.Value{inputTensor},
				[]ort.Value{outputTensor},
				opts,
			)
			if err != nil {
				fmt.Printf("  创建 Session 失败: %v\n", err)
				return
			}
			defer session.Destroy()

			batchSize := numRequests / concurrency
			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				start := time.Now()
				err := session.Run()
				latency := float64(time.Since(start).Milliseconds())

				if err != nil {
					errors <- err
				} else {
					latencies <- latency
				}
			}
		}()
	}

	wg.Wait()
	close(latencies)
	close(errors)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	latencyList := make([]float64, 0, numRequests)
	for lat := range latencies {
		latencyList = append(latencyList, lat)
	}

	errorCount := 0
	for range errors {
		errorCount++
	}

	if len(latencyList) == 0 {
		return &ArchitectureTestResult{
			Architecture:   "Session Pool",
			Concurrency:    concurrency,
			TotalRequests:  numRequests,
			FailedRequests: numRequests,
			TotalTime:      totalTime,
		}
	}

	sort.Float64s(latencyList)

	sumLatency := 0.0
	for _, lat := range latencyList {
		sumLatency += lat
	}

	return &ArchitectureTestResult{
		Architecture:       "Session Pool",
		Concurrency:        concurrency,
		TotalRequests:      numRequests,
		SuccessfulRequests: len(latencyList),
		FailedRequests:     errorCount,
		TotalTime:          totalTime,
		AvgLatency:         sumLatency / float64(len(latencyList)),
		P50Latency:         calculatePercentile(latencyList, 0.50),
		P90Latency:         calculatePercentile(latencyList, 0.90),
		P99Latency:         calculatePercentile(latencyList, 0.99),
		MinLatency:         latencyList[0],
		MaxLatency:         latencyList[len(latencyList)-1],
		Throughput:         float64(len(latencyList)) / (totalTime / 1000.0),
		StartRSS:           startRSS,
		PeakRSS:            peakRSS,
		EndRSS:             endRSS,
		RSSDrift:           endRSS - startRSS,
	}
}

func main() {
	fmt.Println("===== Go 并发推理架构对比测试（学术版）=====")
	fmt.Println()
	fmt.Println("研究问题：在高并发推理场景下，共享 InferenceSession 会产生资源竞争，")
	fmt.Println("          从而影响系统吞吐量和延迟稳定性。")
	fmt.Println()
	fmt.Println("测试三种架构：")
	fmt.Println("  1. Shared Session - 多线程共享同一个 Session")
	fmt.Println("  2. Mutex          - 串行化访问共享 Session")
	fmt.Println("  3. Session Pool   - 每个 goroutine 独立 Session")
	fmt.Println()

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化 ONNX Runtime 环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return
	}

	runtime.GOMAXPROCS(12)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	concurrencyLevels := []int{1, 2, 4, 6, 8, 12}
	numRequests := 500

	allResults := make(map[string][]*ArchitectureTestResult)
	allResults["Shared Session"] = make([]*ArchitectureTestResult, 0)
	allResults["Mutex"] = make([]*ArchitectureTestResult, 0)
	allResults["Session Pool"] = make([]*ArchitectureTestResult, 0)

	// 测试三种架构
	for _, concurrency := range concurrencyLevels {
		fmt.Printf("\n===== 并发度: %d =====\n\n", concurrency)

		// 1. Shared Session
		result1 := runSharedSessionTest(modelPath, inputData, inputShape, outputShape, concurrency, numRequests)
		if result1 != nil {
			allResults["Shared Session"] = append(allResults["Shared Session"], result1)
			fmt.Printf("    吞吐量: %.2f REQ/s, P99: %.2f ms\n", result1.Throughput, result1.P99Latency)
		}

		// 2. Mutex
		result2 := runMutexTest(modelPath, inputData, inputShape, outputShape, concurrency, numRequests)
		if result2 != nil {
			allResults["Mutex"] = append(allResults["Mutex"], result2)
			fmt.Printf("    吞吐量: %.2f REQ/s, P99: %.2f ms\n", result2.Throughput, result2.P99Latency)
		}

		// 3. Session Pool
		result3 := runSessionPoolTest(modelPath, inputData, inputShape, outputShape, concurrency, numRequests)
		if result3 != nil {
			allResults["Session Pool"] = append(allResults["Session Pool"], result3)
			fmt.Printf("    吞吐量: %.2f REQ/s, P99: %.2f ms\n", result3.Throughput, result3.P99Latency)
		}
	}

	// 保存结果
	resultPath := filepath.Join(basePath, "results", "go_concurrent_architecture_comparison.txt")
	os.MkdirAll(filepath.Dir(resultPath), 0755)

	resultContent := "===== Go 并发推理架构对比测试结果（学术版）=====\n\n"
	resultContent += "研究问题：在高并发推理场景下，共享 InferenceSession 会产生资源竞争，\n"
	resultContent += "          从而影响系统吞吐量和延迟稳定性。\n\n"
	resultContent += "测试三种架构：\n"
	resultContent += "  1. Shared Session - 多线程共享同一个 Session\n"
	resultContent += "  2. Mutex          - 串行化访问共享 Session\n"
	resultContent += "  3. Session Pool   - 每个 goroutine 独立 Session\n\n"

	for arch, results := range allResults {
		resultContent += fmt.Sprintf("===== %s =====\n\n", arch)
		for _, r := range results {
			resultContent += fmt.Sprintf("并发度: %d\n", r.Concurrency)
			resultContent += fmt.Sprintf("  吞吐量: %.2f REQ/s\n", r.Throughput)
			resultContent += fmt.Sprintf("  平均延迟: %.2f ms\n", r.AvgLatency)
			resultContent += fmt.Sprintf("  P50延迟: %.2f ms\n", r.P50Latency)
			resultContent += fmt.Sprintf("  P90延迟: %.2f ms\n", r.P90Latency)
			resultContent += fmt.Sprintf("  P99延迟: %.2f ms\n", r.P99Latency)
			resultContent += fmt.Sprintf("  峰值RSS: %.2f MB\n\n", r.PeakRSS)
		}
	}

	err = os.WriteFile(resultPath, []byte(resultContent), 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		return
	}

	fmt.Printf("\n结果已保存到: %s\n", resultPath)
	fmt.Println("\n===== 测试完成 =====")
}
