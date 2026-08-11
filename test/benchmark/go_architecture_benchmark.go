// go_architecture_benchmark.go
// Go 并发推理架构对比测试（三种架构：Unsafe Shared / Mutex / Session Pool）
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置动态 intraOp（测试时 intraOp=1）
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 每种架构在不同并发度（1/2/4/8/12）下测试吞吐量、延迟、内存
//
// 测试目的：
// - 对比三种并发推理架构的性能特征
// - 验证 Session Pool 架构在并发场景下的优势
// - 为论文架构对比实验提供数据

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

type Architecture int

const (
	SharedSession Architecture = iota
	MutexProtected
	SessionPool
)

func (a Architecture) String() string {
	return []string{"Unsafe Shared", "Mutex Shared", "Session Pool"}[a]
}

type TestResult struct {
	Architecture       Architecture
	Concurrency        int
	PoolSize           int
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

// sessionWithTensors 包含 session 和绑定的 tensor
type sessionWithTensors struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputData    []byte // 原始输入数据
}

// createSessionWithTensors 创建 Session 并绑定专用 tensor
func createSessionWithTensors(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	intraOpThreads int,
) (*sessionWithTensors, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建会话选项失败：%v", err)
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入 Tensor 失败：%v", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出 Tensor 失败：%v", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		opts,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建 Session 失败：%v", err)
	}

	return &sessionWithTensors{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputData:    inputData,
	}, nil
}

// fillInputData 填充输入数据到 tensor
func fillInputData(inputTensor *ort.Tensor[float32], inputData []byte) {
	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

// runInference 执行推理
func runInference(swt *sessionWithTensors) error {
	fillInputData(swt.inputTensor, swt.inputData)
	return swt.session.Run()
}

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

// testUnsafeShared 测试 Unsafe Shared 架构（Single-Session Concurrent）
// 共享单个 Session（ORT Run 线程安全），每个 goroutine 持有独立的 I/O Tensor
// 避免在共享单 Session 时复用同一组 I/O OrtValue 导致逻辑冲突。
// 注意：本实现仍共享 Session 的 CPU 内存分配器（Arena）与线程池，
// 高并发下 Arena 的高水位保持与碎片化会导致 RSS 漂移（见论文 4.1 节）。
func testUnsafeShared(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
	intraOpThreads int,
) *TestResult {
	fmt.Printf("测试 Unsafe Shared: %d 并发，%d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	// 创建共享 session（使用临时 tensor 初始化）
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return &TestResult{FailedRequests: numRequests}
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	// 创建临时 tensor 用于初始化 session
	tempInput, _ := ort.NewEmptyTensor[float32](inputShape)
	defer tempInput.Destroy()
	tempOutput, _ := ort.NewEmptyTensor[float32](outputShape)
	defer tempOutput.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{tempInput},
		[]ort.ArbitraryTensor{tempOutput},
		opts,
	)
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return &TestResult{FailedRequests: numRequests}
	}
	defer session.Destroy()

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// 每个 goroutine 创建独立的 tensor
			inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
			if err != nil {
				for j := 0; j < batchSize; j++ {
					errorChan <- err
				}
				return
			}
			defer inputTensor.Destroy()

			outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
			if err != nil {
				for j := 0; j < batchSize; j++ {
					errorChan <- err
				}
				return
			}
			defer outputTensor.Destroy()

			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				// 填充数据
				fillInputData(inputTensor, inputData)

				// 运行推理（共享 Session，ORT Run 线程安全；各 goroutine 独立 Tensor）
				start := time.Now()
				err := session.Run()
				latency := float64(time.Since(start).Milliseconds())

				if err != nil {
					errorChan <- err
				} else {
					latencyChan <- latency
				}
			}
		}(i)
	}

	wg.Wait()
	close(latencyChan)
	close(errorChan)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	return collectResults(numRequests, totalTime, startRSS, peakRSS, endRSS, latencyChan, errorChan)
}

// testMutexShared 测试 Mutex Shared 架构
// 共享 session 和 tensor，加锁串行化访问
func testMutexShared(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
	intraOpThreads int,
) *TestResult {
	fmt.Printf("测试 Mutex Shared: %d 并发，%d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	swt, err := createSessionWithTensors(modelPath, inputData, inputShape, outputShape, intraOpThreads)
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return &TestResult{FailedRequests: numRequests}
	}
	defer func() {
		swt.session.Destroy()
		swt.inputTensor.Destroy()
		swt.outputTensor.Destroy()
	}()

	var mu sync.Mutex
	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				mu.Lock()
				start := time.Now()
				err := runInference(swt)
				latency := float64(time.Since(start).Milliseconds())
				mu.Unlock()

				if err != nil {
					errorChan <- err
				} else {
					latencyChan <- latency
				}
			}
		}(i)
	}

	wg.Wait()
	close(latencyChan)
	close(errorChan)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	return collectResults(numRequests, totalTime, startRSS, peakRSS, endRSS, latencyChan, errorChan)
}

// testSessionPool 测试 Session Pool 架构
// 每个 worker 有独立的 session 和 tensor（最佳方案）
func testSessionPool(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	poolSize int,
	numRequests int,
	intraOpThreads int,
) *TestResult {
	fmt.Printf("测试 Session Pool: pool_size=%d, %d 请求\n", poolSize, numRequests)

	startRSS := getProcessRSS()
	var peakRSS float64 = startRSS
	var mu sync.Mutex

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	batchSize := numRequests / poolSize

	for i := 0; i < poolSize; i++ {
		swt, err := createSessionWithTensors(modelPath, inputData, inputShape, outputShape, intraOpThreads)
		if err != nil {
			fmt.Printf("  创建 Session %d 失败: %v\n", i, err)
			continue
		}

		wg.Add(1)

		go func(swt *sessionWithTensors) {
			defer wg.Done()
			defer swt.session.Destroy()
			defer swt.inputTensor.Destroy()
			defer swt.outputTensor.Destroy()

			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				mu.Lock()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
				mu.Unlock()

				start := time.Now()
				err := runInference(swt)
				latency := float64(time.Since(start).Milliseconds())

				if err != nil {
					errorChan <- err
				} else {
					latencyChan <- latency
				}
			}
		}(swt)
	}

	wg.Wait()
	close(latencyChan)
	close(errorChan)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	mu.Lock()
	finalPeakRSS := peakRSS
	mu.Unlock()

	result := collectResults(numRequests, totalTime, startRSS, finalPeakRSS, endRSS, latencyChan, errorChan)
	result.PoolSize = poolSize
	return result
}

func collectResults(
	numRequests int,
	totalTime float64,
	startRSS, peakRSS, endRSS float64,
	latencyChan <-chan float64,
	errorChan <-chan error,
) *TestResult {
	latencies := make([]float64, 0, numRequests)
	for lat := range latencyChan {
		latencies = append(latencies, lat)
	}

	errorCount := 0
	for range errorChan {
		errorCount++
	}

	if len(latencies) == 0 {
		return &TestResult{
			TotalRequests:  numRequests,
			FailedRequests: numRequests,
			TotalTime:      totalTime,
			StartRSS:       startRSS,
			PeakRSS:        peakRSS,
			EndRSS:         endRSS,
			RSSDrift:       endRSS - startRSS,
		}
	}

	sort.Float64s(latencies)

	sumLatency := 0.0
	minLatency := latencies[0]
	maxLatency := latencies[len(latencies)-1]

	for _, latency := range latencies {
		sumLatency += latency
	}

	avgLatency := sumLatency / float64(len(latencies))
	p50 := calculatePercentile(latencies, 0.50)
	p90 := calculatePercentile(latencies, 0.90)
	p99 := calculatePercentile(latencies, 0.99)

	throughput := float64(len(latencies)) / (totalTime / 1000.0)

	return &TestResult{
		TotalRequests:      numRequests,
		SuccessfulRequests: len(latencies),
		FailedRequests:     errorCount,
		TotalTime:          totalTime,
		AvgLatency:         avgLatency,
		P50Latency:         p50,
		P90Latency:         p90,
		P99Latency:         p99,
		MinLatency:         minLatency,
		MaxLatency:         maxLatency,
		Throughput:         throughput,
		StartRSS:           startRSS,
		PeakRSS:            peakRSS,
		EndRSS:             endRSS,
		RSSDrift:           endRSS - startRSS,
	}
}

func main() {
	fmt.Println("===== Go 推理架构性能对比实验（论文级）=====")

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败：%v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	fmt.Printf("库路径: %s\n", libPath)
	fmt.Printf("模型路径: %s\n", modelPath)
	fmt.Printf("输入数据路径: %s\n", inputDataPath)

	// 检查文件是否存在
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		fmt.Printf("错误: 库文件不存在: %s\n", libPath)
		os.Exit(1)
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("错误: 模型文件不存在: %s\n", modelPath)
		os.Exit(1)
	}
	if _, err := os.Stat(inputDataPath); os.IsNotExist(err) {
		fmt.Printf("错误: 输入数据文件不存在: %s\n", inputDataPath)
		os.Exit(1)
	}

	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化 ONNX Runtime 环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()
	fmt.Println("ONNX Runtime 环境初始化成功")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败：%v\n", err)
		return
	}
	fmt.Printf("输入数据大小: %d bytes\n", len(inputData))

	runtime.GOMAXPROCS(12)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	allResults := make([]*TestResult, 0)

	fmt.Println("\n===== 实验 1: Unsafe Shared 扩展性测试 =====")
	for _, concurrency := range []int{1, 2, 4, 8, 12} {
		result := testUnsafeShared(modelPath, inputData, inputShape, outputShape, concurrency, 500, 1)
		result.Architecture = SharedSession
		result.Concurrency = concurrency
		allResults = append(allResults, result)

		// 控制台输出保留2位小数（便于阅读），文件保存保留5位小数
		fmt.Printf("并发=%d, 吞吐量=%.2f REQ/s, 平均延迟=%.2f ms, P99=%.2f ms\n",
			concurrency, result.Throughput, result.AvgLatency, result.P99Latency)
	}

	fmt.Println("\n===== 实验 2: Mutex Shared 串行化测试 =====")
	for _, concurrency := range []int{1, 2, 4, 8, 12} {
		result := testMutexShared(modelPath, inputData, inputShape, outputShape, concurrency, 500, 1)
		result.Architecture = MutexProtected
		result.Concurrency = concurrency
		allResults = append(allResults, result)

		// 控制台输出保留2位小数（便于阅读），文件保存保留5位小数
		fmt.Printf("并发=%d, 吞吐量=%.2f REQ/s, 平均延迟=%.2f ms, P99=%.2f ms\n",
			concurrency, result.Throughput, result.AvgLatency, result.P99Latency)
	}

	fmt.Println("\n===== 实验 3: Session Pool 池大小优化测试 =====")
	for _, poolSize := range []int{1, 2, 4, 6, 8, 12} {
		result := testSessionPool(modelPath, inputData, inputShape, outputShape, poolSize, 500, 1)
		result.Architecture = SessionPool
		result.PoolSize = poolSize
		allResults = append(allResults, result)

		fmt.Printf("池大小=%d, 吞吐量=%.2f REQ/s, 平均延迟=%.2f ms, P99=%.2f ms\n",
			poolSize, result.Throughput, result.AvgLatency, result.P99Latency)
	}

	resultPath := filepath.Join(basePath, "results", "go_architecture_comparison.txt")
	os.MkdirAll(filepath.Dir(resultPath), 0755)

	var content strings.Builder
	content.WriteString("===== Go 推理架构性能对比实验结果 =====\n\n")
	content.WriteString("测试三种架构：\n")
	content.WriteString("  1. Unsafe Shared - 共享 Session，独立 Tensor（测试 Session Contention）\n")
	content.WriteString("  2. Mutex Shared  - 共享 Session，加锁串行化\n")
	content.WriteString("  3. Session Pool  - 独立 Session（最佳方案）\n\n")

	for _, r := range allResults {
		content.WriteString(fmt.Sprintf("===== %s =====\n", r.Architecture))
		if r.Architecture == SessionPool {
			content.WriteString(fmt.Sprintf("池大小: %d\n", r.PoolSize))
		} else {
			content.WriteString(fmt.Sprintf("并发度: %d\n", r.Concurrency))
		}
		// 中间数据保留5位小数，符合核心期刊规范
		content.WriteString(fmt.Sprintf("  吞吐量: %.5f REQ/s\n", r.Throughput))
		content.WriteString(fmt.Sprintf("  平均延迟: %.5f ms\n", r.AvgLatency))
		content.WriteString(fmt.Sprintf("  P50延迟: %.5f ms\n", r.P50Latency))
		content.WriteString(fmt.Sprintf("  P90延迟: %.5f ms\n", r.P90Latency))
		content.WriteString(fmt.Sprintf("  P99延迟: %.5f ms\n", r.P99Latency))
		content.WriteString(fmt.Sprintf("  最小延迟: %.5f ms\n", r.MinLatency))
		content.WriteString(fmt.Sprintf("  最大延迟: %.5f ms\n", r.MaxLatency))
		content.WriteString(fmt.Sprintf("  峰值PM: %.5f MB\n", r.PeakRSS))
		content.WriteString(fmt.Sprintf("  PM漂移: %.5f MB\n\n", r.RSSDrift))
	}

	err = os.WriteFile(resultPath, []byte(content.String()), 0644)
	if err != nil {
		fmt.Printf("保存结果失败：%v\n", err)
		return
	}

	fmt.Printf("\n结果已保存到：%s\n", resultPath)
	fmt.Println("\n===== 实验完成 =====")
}
