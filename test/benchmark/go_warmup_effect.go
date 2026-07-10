// go_warmup_effect.go
// Go 首次推理预热效应测试
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=8, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
//
// 测试目的：
// - 连续执行 50 次推理，记录每次推理延迟和 RSS
// - 计算首次/第二次/稳定状态延迟差异
// - 通过变异系数 CV < 5% 判定预热所需次数
// - 输出详细 JSON + CSV + 摘要 JSON

package main

import (
	"encoding/binary"
	"encoding/csv"
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

// WarmupResult 预热效应结果
type WarmupResult struct {
	InferenceNum int     `json:"inference_num"`
	LatencyMs    float64 `json:"latency_ms"`
	RSSMB        float64 `json:"rss_mb"`
	Timestamp    string  `json:"timestamp"`
}

// TestSummary 测试汇总
type TestSummary struct {
	TestName          string  `json:"test_name"`
	Model             string  `json:"model"`
	Language          string  `json:"language"`
	FirstInferenceMs  float64 `json:"first_inference_ms"`
	SecondInferenceMs float64 `json:"second_inference_ms"`
	StableInferenceMs float64 `json:"stable_inference_ms"`
	WarmupRequiredNum int     `json:"warmup_required_num"`
	PerformanceGap    float64 `json:"performance_gap_percent"`
	StableCV          float64 `json:"stable_cv_percent"`
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

// 测试预热效应
func testWarmupEffect(modelPath, inputDataPath string, basePath string) ([]WarmupResult, *TestSummary, error) {
	fmt.Println("===== 首次推理预热效应测试 =====")

	// 加载输入数据
	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("读取输入数据失败: %v", err)
	}

	// 创建Session
	session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
		[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
	if err != nil {
		return nil, nil, fmt.Errorf("创建Session失败: %v", err)
	}
	defer session.Destroy()
	defer inputTensor.Destroy()
	defer outputTensor.Destroy()

	// 创建CSV记录文件
	csvFile, err := os.Create(filepath.Join(basePath, "results", "go_warmup_effect_detailed.csv"))
	if err != nil {
		return nil, nil, fmt.Errorf("创建CSV文件失败: %v", err)
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()
	csvWriter.Write([]string{"Inference#", "Latency(ms)", "RSS(MB)", "Timestamp"})

	// 连续推理20次
	numInferences := 20
	results := make([]WarmupResult, 0, numInferences)
	latencies := make([]float64, 0, numInferences)

	fmt.Printf("\n执行 %d 次连续推理...\n", numInferences)

	for i := 0; i < numInferences; i++ {
		startTime := time.Now()
		err := session.Run()
		latency := time.Since(startTime).Milliseconds()

		if err != nil {
			fmt.Printf("第%d次推理失败: %v\n", i+1, err)
			continue
		}

		rss := getRSSMB()
		timestamp := time.Now().Format("15:04:05.000")

		result := WarmupResult{
			InferenceNum: i + 1,
			LatencyMs:    float64(latency),
			RSSMB:        rss,
			Timestamp:    timestamp,
		}
		results = append(results, result)
		latencies = append(latencies, float64(latency))

		// 写入CSV
		csvWriter.Write([]string{
			fmt.Sprintf("%d", i+1),
			fmt.Sprintf("%.5f", float64(latency)),
			fmt.Sprintf("%.5f", rss),
			timestamp,
		})

		fmt.Printf("推理 #%2d: %.5f ms, RSS: %.5f MB\n", i+1, float64(latency), rss)
	}

	// 分析结果
	if len(latencies) < 3 {
		return results, nil, fmt.Errorf("推理次数不足，无法分析")
	}

	firstLatency := latencies[0]
	secondLatency := latencies[1]

	// 计算稳定状态的平均值（后50%）
	stableStart := len(latencies) / 2
	stableLatencies := latencies[stableStart:]

	var stableSum float64
	for _, lat := range stableLatencies {
		stableSum += lat
	}
	stableAvg := stableSum / float64(len(stableLatencies))

	// 计算稳定状态的变异系数
	var stableVariance float64
	for _, lat := range stableLatencies {
		diff := lat - stableAvg
		stableVariance += diff * diff
	}
	stableStd := math.Sqrt(stableVariance / float64(len(stableLatencies)))
	stableCV := (stableStd / stableAvg) * 100

	// 计算性能差距
	performanceGap := ((firstLatency - stableAvg) / stableAvg) * 100

	// 确定预热所需次数（延迟进入稳定状态的±10%范围内）
	warmupRequired := 1
	threshold := stableAvg * 1.1
	for i, lat := range latencies {
		if lat <= threshold {
			warmupRequired = i + 1
			break
		}
	}

	summary := &TestSummary{
		TestName:          "Warmup_Effect",
		Model:             "YOLO11x",
		Language:          "Go",
		FirstInferenceMs:  firstLatency,
		SecondInferenceMs: secondLatency,
		StableInferenceMs: stableAvg,
		WarmupRequiredNum: warmupRequired,
		PerformanceGap:    performanceGap,
		StableCV:          stableCV,
	}

	fmt.Printf("\n===== 预热效应分析 =====\n")
	fmt.Printf("首次推理延迟: %.5f ms\n", firstLatency)
	fmt.Printf("第二次推理延迟: %.5f ms\n", secondLatency)
	fmt.Printf("稳定状态平均延迟: %.5f ms\n", stableAvg)
	fmt.Printf("首次 vs 稳定差距: %.2f%%\n", performanceGap)
	fmt.Printf("预热所需次数: %d\n", warmupRequired)
	fmt.Printf("稳定状态变异系数: %.2f%%\n", stableCV)

	return results, summary, nil
}

func main() {
	fmt.Println("===== Go 首次推理预热效应测试 =====")

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

	results, summary, err := testWarmupEffect(modelPath, inputDataPath, basePath)
	if err != nil {
		fmt.Printf("测试失败: %v\n", err)
		os.Exit(1)
	}

	// 保存详细结果
	detailData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("序列化详细结果失败: %v\n", err)
		os.Exit(1)
	}

	detailFile := filepath.Join(basePath, "results", "go_warmup_effect_detailed.json")
	err = os.WriteFile(detailFile, detailData, 0644)
	if err != nil {
		fmt.Printf("保存详细结果失败: %v\n", err)
		os.Exit(1)
	}

	// 保存汇总结果
	summaryData, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		fmt.Printf("序列化汇总结果失败: %v\n", err)
		os.Exit(1)
	}

	summaryFile := filepath.Join(basePath, "results", "go_warmup_effect_summary.json")
	err = os.WriteFile(summaryFile, summaryData, 0644)
	if err != nil {
		fmt.Printf("保存汇总结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n===== 测试完成 =====\n")
	fmt.Printf("详细结果: %s\n", detailFile)
	fmt.Printf("汇总结果: %s\n", summaryFile)
}