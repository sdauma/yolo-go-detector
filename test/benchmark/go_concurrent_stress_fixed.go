// go_concurrent_stress_fixed.go
// Go Session Pool 并发推理性能测试（修复版）
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=1, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 关键修复：每个 goroutine 拥有独立的 Session 和绑定的 Tensor，循环复用
// - 测试并发度 1/2/4/6/8/12，每并发度 500 次请求
//
// 测试目的：
// - 验证 Session Pool 在修复后的正确并发行为
// - 测量独立 Session 复用模式下的性能表现

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

type ConcurrentTestResult struct {
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

// SessionBundle 包含Session和绑定的Tensor
type SessionBundle struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

// createSessionBundle 创建Session和绑定的Tensor
// 关键：Tensor在Session创建时绑定，之后循环复用，不能每次推理重新创建
func createSessionBundle(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	intraOpThreads int,
) *SessionBundle {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("创建会话选项失败：%v\n", err)
		return nil
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	// 创建输入Tensor - 这个Tensor将和Session绑定，循环复用
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入Tensor失败：%v\n", err)
		return nil
	}

	// 创建输出Tensor - 这个Tensor将和Session绑定，循环复用
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		fmt.Printf("创建输出Tensor失败：%v\n", err)
		return nil
	}

	// 创建Session，绑定inputTensor和outputTensor
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		opts,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		fmt.Printf("创建Session失败：%v\n", err)
		return nil
	}

	return &SessionBundle{
		session: session,
		input:   inputTensor,
		output:  outputTensor,
	}
}

// runConcurrentInference 执行推理
// 关键：使用bundle中已绑定的Tensor，只修改数据，不创建新Tensor
func runConcurrentInference(
	bundle *SessionBundle,
	inputData []byte,
	latencies chan<- float64,
	errors chan<- error,
) {
	// 获取输入Tensor的数据缓冲区，直接修改数据
	floatData := bundle.input.GetData()

	// 填充输入数据
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	// 执行推理 - 使用已绑定的Tensor
	startTime := time.Now()
	err := bundle.session.Run()
	latency := float64(time.Since(startTime).Milliseconds())

	if err != nil {
		errors <- err
		return
	}

	latencies <- latency
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

func runConcurrentTest(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
	intraOpThreads int,
) *ConcurrentTestResult {
	fmt.Printf("执行并发测试：%d 并发，%d 请求，intra_op_threads=%d\n",
		concurrency, numRequests, intraOpThreads)

	startRSS := getProcessRSS()
	var peakRSS float64 = startRSS
	var mu sync.Mutex

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()

	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		// 每个goroutine创建独立的SessionBundle（包含Session和绑定的Tensor）
		bundle := createSessionBundle(modelPath, inputData, inputShape, outputShape, intraOpThreads)

		if bundle == nil {
			fmt.Printf("创建 SessionBundle %d 失败\n", i)
			continue
		}

		wg.Add(1)

		go func(b *SessionBundle) {
			// 关键：goroutine结束时销毁Session和Tensor
			defer wg.Done()
			defer b.session.Destroy()
			defer b.input.Destroy()
			defer b.output.Destroy()

			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				mu.Lock()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
				mu.Unlock()

				// 使用绑定的Tensor执行推理
				runConcurrentInference(b, inputData, latencyChan, errorChan)
			}
		}(bundle)
	}

	wg.Wait()
	close(latencyChan)
	close(errorChan)

	totalTime := float64(time.Since(startTime).Milliseconds())
	endRSS := getProcessRSS()

	latencies := make([]float64, 0, numRequests)
	for lat := range latencyChan {
		latencies = append(latencies, lat)
	}

	errorCount := 0
	for range errorChan {
		errorCount++
	}

	if len(latencies) == 0 {
		return &ConcurrentTestResult{
			TotalRequests:  numRequests,
			FailedRequests: numRequests,
			TotalTime:      totalTime,
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

	mu.Lock()
	finalPeakRSS := peakRSS
	mu.Unlock()

	return &ConcurrentTestResult{
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
		PeakRSS:            finalPeakRSS,
		EndRSS:             endRSS,
		RSSDrift:           endRSS - startRSS,
	}
}

func main() {
	fmt.Println("===== Go Session Pool 并发推理性能测试（学术版）=====")
	fmt.Println("关键修复：每个goroutine拥有独立的Session和绑定的Tensor，循环复用")

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
		fmt.Printf("初始化环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败：%v\n", err)
		return
	}

	runtime.GOMAXPROCS(12)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	concurrencyLevels := []int{1, 2, 4, 6, 8, 12}
	numRequests := 500

	fmt.Println("\n===== Session Pool 扩展性测试（并发度 vs CPU 核心数）=====")
	results := make([]*ConcurrentTestResult, 0, len(concurrencyLevels))

	for _, concurrency := range concurrencyLevels {
		fmt.Printf("\n===== 测试配置：%d 并发 =====\n", concurrency)
		result := runConcurrentTest(modelPath, inputData, inputShape, outputShape, concurrency, numRequests, 1)
		results = append(results, result)

		fmt.Printf("总请求数：%d\n", result.TotalRequests)
		fmt.Printf("成功请求数：%d\n", result.SuccessfulRequests)
		fmt.Printf("失败请求数：%d\n", result.FailedRequests)
		fmt.Printf("总时间：%.5f ms\n", result.TotalTime)
		fmt.Printf("平均延迟：%.5f ms\n", result.AvgLatency)
		fmt.Printf("P50延迟：%.5f ms\n", result.P50Latency)
		fmt.Printf("P90延迟：%.5f ms\n", result.P90Latency)
		fmt.Printf("P99延迟：%.5f ms\n", result.P99Latency)
		fmt.Printf("最小延迟：%.5f ms\n", result.MinLatency)
		fmt.Printf("最大延迟：%.5f ms\n", result.MaxLatency)
		fmt.Printf("吞吐量：%.5f REQ/s\n", result.Throughput)
		fmt.Printf("初始PM：%.5f MB\n", result.StartRSS)
		fmt.Printf("峰值PM：%.5f MB\n", result.PeakRSS)
		fmt.Printf("最终RSS：%.5f MB\n", result.EndRSS)
		fmt.Printf("PM漂移：%.5f MB\n", result.RSSDrift)
	}

	resultPath := filepath.Join(basePath, "results", "go_session_pool_performance.txt")
	os.MkdirAll(filepath.Dir(resultPath), 0755)

	resultContent := "===== Go Session Pool 并发推理性能测试结果（学术版）=====\n"
	resultContent += "关键修复：每个goroutine拥有独立的Session和绑定的Tensor，循环复用\n\n"
	for i, result := range results {
		concurrency := concurrencyLevels[i]
		resultContent += fmt.Sprintf("===== 测试配置：%d 并发 =====\n", concurrency)
		resultContent += fmt.Sprintf("总请求数：%d\n", result.TotalRequests)
		resultContent += fmt.Sprintf("成功请求数：%d\n", result.SuccessfulRequests)
		resultContent += fmt.Sprintf("失败请求数：%d\n", result.FailedRequests)
		resultContent += fmt.Sprintf("总时间：%.5f ms\n", result.TotalTime)
		resultContent += fmt.Sprintf("平均延迟：%.5f ms\n", result.AvgLatency)
		resultContent += fmt.Sprintf("P50延迟：%.5f ms\n", result.P50Latency)
		resultContent += fmt.Sprintf("P90延迟：%.5f ms\n", result.P90Latency)
		resultContent += fmt.Sprintf("P99延迟：%.5f ms\n", result.P99Latency)
		resultContent += fmt.Sprintf("最小延迟：%.5f ms\n", result.MinLatency)
		resultContent += fmt.Sprintf("最大延迟：%.5f ms\n", result.MaxLatency)
		resultContent += fmt.Sprintf("吞吐量：%.5f REQ/s\n", result.Throughput)
		resultContent += fmt.Sprintf("初始PM：%.5f MB\n", result.StartRSS)
		resultContent += fmt.Sprintf("峰值PM：%.5f MB\n", result.PeakRSS)
		resultContent += fmt.Sprintf("最终RSS：%.5f MB\n", result.EndRSS)
		resultContent += fmt.Sprintf("PM漂移：%.5f MB\n\n", result.RSSDrift)
	}

	err = os.WriteFile(resultPath, []byte(resultContent), 0644)
	if err != nil {
		fmt.Printf("保存结果失败：%v\n", err)
		return
	}

	fmt.Printf("\n结果已保存到：%s\n", resultPath)
	fmt.Println("\n===== 测试完成 =====")
}
