package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type TestResult struct {
	Architecture       string
	Concurrency        int
	TotalRequests      int
	SuccessfulRequests int
	FailedRequests     int
	TotalTime          float64
	AvgLatency         float64
	P99Latency         float64
	Throughput         float64
	PeakRSS            float64
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

// createSessionOnly 只创建 Session，不绑定 tensor
func createSessionOnly(modelPath string, intraOpThreads int) (*ort.AdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建会话选项失败：%v", err)
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	tempInput, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建临时输入 Tensor 失败：%v", err)
	}
	defer tempInput.Destroy()

	tempOutput, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("创建临时输出 Tensor 失败：%v", err)
	}
	defer tempOutput.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{tempInput},
		[]ort.Value{tempOutput},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("创建 Session 失败：%v", err)
	}

	return session, nil
}

// runInferenceWithNewTensors 为每次推理创建新的 tensors
func runInferenceWithNewTensors(
	session *ort.AdvancedSession,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
) error {
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return fmt.Errorf("创建输入 Tensor 失败：%v", err)
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
		return fmt.Errorf("创建输出 Tensor 失败：%v", err)
	}
	defer outputTensor.Destroy()

	return session.Run()
}

func testSharedSession(modelPath string, inputData []byte, inputShape, outputShape []int64, concurrency, numRequests int) *TestResult {
	fmt.Printf("测试 Shared Session: %d 并发，%d 请求\n", concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	session, err := createSessionOnly(modelPath, 1)
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
			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				start := time.Now()
				err := runInferenceWithNewTensors(session, inputData, inputShape, outputShape)
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
			Architecture:   "Shared",
			Concurrency:    concurrency,
			FailedRequests: numRequests,
		}
	}

	sort.Float64s(latencies)

	sumLatency := 0.0
	for _, latency := range latencies {
		sumLatency += latency
	}

	avgLatency := sumLatency / float64(len(latencies))
	p99 := latencies[int(float64(len(latencies))*0.99)]
	throughput := float64(len(latencies)) / (totalTime / 1000.0)

	return &TestResult{
		Architecture:       "Shared",
		Concurrency:        concurrency,
		TotalRequests:      numRequests,
		SuccessfulRequests: len(latencies),
		FailedRequests:     errorCount,
		TotalTime:          totalTime,
		AvgLatency:         avgLatency,
		P99Latency:         p99,
		Throughput:         throughput,
		PeakRSS:            peakRSS,
	}
}

func main() {
	fmt.Println("===== Go 架构对比快速测试 =====")

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
	fmt.Printf("输入数据大小: %d bytes\n\n", len(inputData))

	runtime.GOMAXPROCS(12)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// 快速测试：只测试 1 和 4 并发，各 50 次请求
	fmt.Println("===== 快速测试: Shared Session =====")
	for _, concurrency := range []int{1, 4} {
		result := testSharedSession(modelPath, inputData, inputShape, outputShape, concurrency, 50)

		fmt.Printf("并发=%d, 成功=%d, 失败=%d\n", concurrency, result.SuccessfulRequests, result.FailedRequests)
		fmt.Printf("  吞吐量=%.5f REQ/s, 平均延迟=%.5f ms, P99=%.5f ms\n",
			result.Throughput, result.AvgLatency, result.P99Latency)
		fmt.Printf("  峰值RSS=%.5f MB\n\n", result.PeakRSS)
	}

	fmt.Println("===== 快速测试完成 =====")
	fmt.Println("\n如果测试成功，说明修复有效！")
	fmt.Println("可以运行完整的 go_architecture_benchmark.exe 进行完整测试。")
}
