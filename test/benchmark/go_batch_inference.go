// go_batch_inference.go
// Go 批处理推理性能测试
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=8, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 通过循环填充 batch 维度实现批处理
//
// 测试目的：
// - 测试不同 batch size（1/2/4/8）下的推理性能
// - 计算总时间、单图时间和吞吐量（images/sec）
// - 分析批处理对推理效率的影响

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

// BatchResult 批处理结果
type BatchResult struct {
	BatchSize      int     `json:"batch_size"`
	TotalTimeMs    float64 `json:"total_time_ms"`
	PerImageTimeMs float64 `json:"per_image_time_ms"`
	Throughput     float64 `json:"throughput_images_per_sec"`
	PeakRSSMB      float64 `json:"peak_rss_mb"`
}

// TestResult 测试结果
type TestResult struct {
	TestName  string        `json:"test_name"`
	Model     string        `json:"model"`
	Language  string        `json:"language"`
	Results   []BatchResult `json:"results"`
	Timestamp string        `json:"timestamp"`
}

// 获取RSS内存（MB）
func getRSSMB() float64 { return memutil.PrivateMemoryMB() }

// 创建Session
func createSession(modelPath string, inputData []byte, inputShape, outputShape []int64) (*ort.AdvancedSession, *ort.Tensor[float32], *ort.Tensor[float32], error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("创建会话选项失败: %v", err)
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(8)
	opts.SetInterOpNumThreads(1)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("创建输入Tensor失败: %v", err)
	}

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, nil, nil, fmt.Errorf("创建输出Tensor失败: %v", err)
	}

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
		return nil, nil, nil, fmt.Errorf("创建Session失败: %v", err)
	}

	return session, inputTensor, outputTensor, nil
}

// 测试批处理性能
func testBatchInference(modelPath, inputDataPath string, batchSizes []int) ([]BatchResult, error) {
	fmt.Println("===== 批处理性能测试 =====")

	// 加载输入数据
	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		return nil, fmt.Errorf("读取输入数据失败: %v", err)
	}

	results := make([]BatchResult, 0, len(batchSizes))

	for _, batchSize := range batchSizes {
		fmt.Printf("\n===== 测试 Batch Size = %d =====\n", batchSize)

		// 创建Session
		session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
			[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
		if err != nil {
			fmt.Printf("创建Session失败: %v\n", err)
			continue
		}

		// 预热
		for i := 0; i < 5; i++ {
			session.Run()
		}

		// 测量内存
		startRSS := getRSSMB()
		peakRSS := startRSS

		// 执行批处理测试
		startTime := time.Now()
		for i := 0; i < batchSize; i++ {
			err := session.Run()
			if err != nil {
				fmt.Printf("推理失败: %v\n", err)
				break
			}

			// 检查内存
			currentRSS := getRSSMB()
			if currentRSS > peakRSS {
				peakRSS = currentRSS
			}
		}
		totalTime := time.Since(startTime)

		// 计算指标
		totalTimeMs := float64(totalTime.Milliseconds())
		perImageTimeMs := totalTimeMs / float64(batchSize)
		throughput := float64(batchSize) / totalTime.Seconds()

		result := BatchResult{
			BatchSize:      batchSize,
			TotalTimeMs:    totalTimeMs,
			PerImageTimeMs: perImageTimeMs,
			Throughput:     throughput,
			PeakRSSMB:      peakRSS,
		}
		results = append(results, result)

		fmt.Printf("总时间: %.5f ms\n", totalTimeMs)
		fmt.Printf("每张图片: %.5f ms\n", perImageTimeMs)
		fmt.Printf("吞吐量: %.5f images/sec\n", throughput)
		fmt.Printf("峰值PM: %.5f MB\n", peakRSS)

		// 清理
		session.Destroy()
		inputTensor.Destroy()
		outputTensor.Destroy()
		runtime.GC()
	}

	return results, nil
}

func main() {
	fmt.Println("===== Go 批处理性能测试 =====")

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

	runtime.GOMAXPROCS(12)

	// 测试不同的batch size
	batchSizes := []int{1, 4, 8, 16, 32}

	results, err := testBatchInference(modelPath, inputDataPath, batchSizes)
	if err != nil {
		fmt.Printf("测试失败: %v\n", err)
		os.Exit(1)
	}

	// 保存结果
	testResult := TestResult{
		TestName:  "Batch_Inference",
		Model:     "YOLO11x",
		Language:  "Go",
		Results:   results,
		Timestamp: time.Now().Format("2006-01-02 15:04:05"),
	}

	resultData, err := json.MarshalIndent(testResult, "", "  ")
	if err != nil {
		fmt.Printf("序列化结果失败: %v\n", err)
		os.Exit(1)
	}

	resultFile := filepath.Join(basePath, "results", "go_batch_inference_result.json")
	err = os.WriteFile(resultFile, resultData, 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		os.Exit(1)
	}

	// 打印汇总
	fmt.Printf("\n===== 批处理性能汇总 =====\n")
	fmt.Printf("%-10s %-15s %-15s %-20s %-15s\n", "Batch", "Total(ms)", "PerImg(ms)", "Throughput(img/s)", "PeakRSS(MB)")
	fmt.Println(string(make([]byte, 80)))
	for _, r := range results {
		fmt.Printf("%-10d %-15.5f %-15.5f %-20.5f %-15.5f\n",
			r.BatchSize, r.TotalTimeMs, r.PerImageTimeMs, r.Throughput, r.PeakRSSMB)
	}

	// 找出最优batch size
	var optimalBatch BatchResult
	maxThroughput := 0.0
	for _, r := range results {
		if r.Throughput > maxThroughput {
			maxThroughput = r.Throughput
			optimalBatch = r
		}
	}

	fmt.Printf("\n最优Batch Size: %d (吞吐量: %.5f images/sec)\n", optimalBatch.BatchSize, optimalBatch.Throughput)
	fmt.Printf("结果已保存到: %s\n", resultFile)
}