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
)

// GCStats GC统计信息
type GCStats struct {
	NumGC        uint32  `json:"num_gc"`
	PauseTotalNs uint64  `json:"pause_total_ns"`
	PauseAvgMs   float64 `json:"pause_avg_ms"`
	HeapAlloc    uint64  `json:"heap_alloc"`
	HeapSys      uint64  `json:"heap_sys"`
	HeapObjects  uint64  `json:"heap_objects"`
	Timestamp    string  `json:"timestamp"`
}

// StabilityResult 稳定性测试结果
type StabilityResult struct {
	TestName           string  `json:"test_name"`
	Model              string  `json:"model"`
	Language           string  `json:"language"`
	Duration           string  `json:"duration"`
	TotalInferences    int     `json:"total_inferences"`
	AvgLatencyMs       float64 `json:"avg_latency_ms"`
	MinLatencyMs       float64 `json:"min_latency_ms"`
	MaxLatencyMs       float64 `json:"max_latency_ms"`
	StdLatencyMs       float64 `json:"std_latency_ms"`
	StartRSSMB         float64 `json:"start_rss_mb"`
	EndRSSMB           float64 `json:"end_rss_mb"`
	RSSDriftMB         float64 `json:"rss_drift_mb"`
	PeakRSSMB          float64 `json:"peak_rss_mb"`
	TotalGCNum         uint32  `json:"total_gc_num"`
	TotalGCPauseMs     float64 `json:"total_gc_pause_ms"`
	AvgGCPauseMs       float64 `json:"avg_gc_pause_ms"`
	MaxGCPauseMs       float64 `json:"max_gc_pause_ms"`
	PerformanceDegrade float64 `json:"performance_degrade_percent"`
	Timestamp          string  `json:"timestamp"`
}

// 测试时长选项
const (
	ShortTest  = 10 * time.Minute
	MediumTest = 1 * time.Hour
	LongTest   = 24 * time.Hour
)

// 获取RSS内存（MB）
func getRSSMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Sys) / 1024 / 1024
}

// 获取GC统计
func getGCStats() GCStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	avgPauseMs := 0.0
	if m.NumGC > 0 {
		avgPauseMs = float64(m.PauseTotalNs) / 1e6 / float64(m.NumGC)
	}

	return GCStats{
		NumGC:        m.NumGC,
		PauseTotalNs: m.PauseTotalNs,
		PauseAvgMs:   avgPauseMs,
		HeapAlloc:    m.HeapAlloc,
		HeapSys:      m.HeapSys,
		HeapObjects:  m.HeapObjects,
		Timestamp:    time.Now().Format("2006-01-02 15:04:05"),
	}
}

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

// 运行长时间稳定性测试
func runLongStabilityTest(modelPath, inputDataPath string, duration time.Duration, basePath string) *StabilityResult {
	fmt.Printf("\n===== 长时间稳定性测试（%v）=====\n", duration)

	// 加载输入数据
	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return nil
	}

	// 创建Session
	session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
		[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
	if err != nil {
		fmt.Printf("创建Session失败: %v\n", err)
		return nil
	}
	defer session.Destroy()
	defer inputTensor.Destroy()
	defer outputTensor.Destroy()

	// 记录初始状态
	startRSS := getRSSMB()
	startGC := getGCStats()
	startTime := time.Now()

	fmt.Printf("开始RSS: %.5f MB\n", startRSS)
	fmt.Printf("初始GC次数: %d\n", startGC.NumGC)

	// 准备数据记录
	latencies := make([]float64, 0)
	gcStats := make([]GCStats, 0)
	peakRSS := startRSS
	totalInferences := 0

	// 创建CSV记录文件
	csvFile, err := os.Create(filepath.Join(basePath, "results", "go_long_stability_detailed.csv"))
	if err != nil {
		fmt.Printf("创建CSV文件失败: %v\n", err)
		return nil
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()
	csvWriter.Write([]string{"Inference#", "Latency(ms)", "RSS(MB)", "GC#", "GC Pause(ms)", "HeapAlloc(MB)", "Timestamp"})

	// 预热
	fmt.Println("预热中...")
	for i := 0; i < 10; i++ {
		session.Run()
	}

	// 主测试循环
	fmt.Printf("开始测试，持续 %v...\n", duration)
	inferenceCount := 0
	checkpoint := 0
	checkpoints := []int{100, 500, 1000, 5000, 10000, 50000, 100000}

	for time.Since(startTime) < duration {
		inferenceStart := time.Now()
		err := session.Run()
		inferenceLatency := time.Since(inferenceStart).Milliseconds()

		if err != nil {
			fmt.Printf("推理失败: %v\n", err)
			continue
		}

		latencies = append(latencies, float64(inferenceLatency))
		totalInferences++
		inferenceCount++

		// 每100次推理记录一次详细数据
		if inferenceCount%100 == 0 {
			currentRSS := getRSSMB()
			currentGC := getGCStats()

			if currentRSS > peakRSS {
				peakRSS = currentRSS
			}

			gcStats = append(gcStats, currentGC)

			// 写入CSV
			csvWriter.Write([]string{
				fmt.Sprintf("%d", totalInferences),
				fmt.Sprintf("%.5f", float64(inferenceLatency)),
				fmt.Sprintf("%.5f", currentRSS),
				fmt.Sprintf("%d", currentGC.NumGC),
				// 中间数据保留5位小数，符合核心期刊规范
				fmt.Sprintf("%.5f", currentGC.PauseAvgMs),
				fmt.Sprintf("%.5f", float64(currentGC.HeapAlloc)/1024/1024),
				time.Now().Format("15:04:05"),
			})

			// 检查点输出
			if checkpoint < len(checkpoints) && totalInferences >= checkpoints[checkpoint] {
				fmt.Printf("检查点 %d 次推理完成，当前RSS: %.5f MB\n", totalInferences, currentRSS)
				checkpoint++
			}
		}

		// 每1000次推理输出一次GC统计
		if totalInferences%1000 == 0 {
			currentGC := getGCStats()
			// 中间数据保留5位小数，符合核心期刊规范
			fmt.Printf("已完成 %d 次推理，GC次数: %d，平均停顿: %.5f ms\n",
				totalInferences, currentGC.NumGC, currentGC.PauseAvgMs)
		}
	}

	// 记录结束状态
	endRSS := getRSSMB()
	endGC := getGCStats()
	endTime := time.Now()

	// 计算统计数据
	var sumLatency, minLatency, maxLatency float64
	minLatency = latencies[0]
	maxLatency = latencies[0]

	for _, lat := range latencies {
		sumLatency += lat
		if lat < minLatency {
			minLatency = lat
		}
		if lat > maxLatency {
			maxLatency = lat
		}
	}

	avgLatency := sumLatency / float64(len(latencies))

	// 计算标准差
	var sumSquares float64
	for _, lat := range latencies {
		diff := lat - avgLatency
		sumSquares += diff * diff
	}
	stdLatency := math.Sqrt(sumSquares / float64(len(latencies)))

	// 计算性能衰减（前10% vs 后10%）
	sampleSize := len(latencies) / 10
	if sampleSize > 0 {
		var earlySum, lateSum float64
		for i := 0; i < sampleSize; i++ {
			earlySum += latencies[i]
			lateSum += latencies[len(latencies)-1-i]
		}
		earlyAvg := earlySum / float64(sampleSize)
		lateAvg := lateSum / float64(sampleSize)
		performanceDegrade := ((lateAvg - earlyAvg) / earlyAvg) * 100

		// 计算GC统计
		gcNum := endGC.NumGC - startGC.NumGC
		gcPauseTotal := float64(endGC.PauseTotalNs-startGC.PauseTotalNs) / 1e6
		avgGCPause := 0.0
		if gcNum > 0 {
			avgGCPause = gcPauseTotal / float64(gcNum)
		}

		result := &StabilityResult{
			TestName:           "Long_Stability_Enhanced",
			Model:              "YOLO11x",
			Language:           "Go",
			Duration:           duration.String(),
			TotalInferences:    totalInferences,
			AvgLatencyMs:       avgLatency,
			MinLatencyMs:       minLatency,
			MaxLatencyMs:       maxLatency,
			StdLatencyMs:       stdLatency,
			StartRSSMB:         startRSS,
			EndRSSMB:           endRSS,
			RSSDriftMB:         endRSS - startRSS,
			PeakRSSMB:          peakRSS,
			TotalGCNum:         gcNum,
			TotalGCPauseMs:     gcPauseTotal,
			AvgGCPauseMs:       avgGCPause,
			MaxGCPauseMs:       0, // 需要额外记录
			PerformanceDegrade: performanceDegrade,
			Timestamp:          endTime.Format("2006-01-02 15:04:05"),
		}

		fmt.Printf("\n===== 测试结果 =====\n")
		fmt.Printf("总推理次数: %d\n", totalInferences)
		fmt.Printf("平均延迟: %.5f ms\n", avgLatency)
		fmt.Printf("延迟范围: %.5f - %.5f ms\n", minLatency, maxLatency)
		fmt.Printf("延迟标准差: %.5f ms\n", stdLatency)
		fmt.Printf("起始RSS: %.5f MB\n", startRSS)
		fmt.Printf("结束RSS: %.5f MB\n", endRSS)
		fmt.Printf("RSS漂移: %.5f MB\n", endRSS-startRSS)
		fmt.Printf("峰值RSS: %.5f MB\n", peakRSS)
		fmt.Printf("GC次数: %d\n", gcNum)
		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Printf("GC总停顿: %.5f ms\n", gcPauseTotal)
		fmt.Printf("平均GC停顿: %.5f ms\n", avgGCPause)
		fmt.Printf("性能衰减: %.2f%%\n", performanceDegrade)

		return result
	}

	return nil
}

func main() {
	fmt.Println("===== Go 长时间稳定性测试（增强版，带GC统计）=====")

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

	// 默认运行10分钟测试
	duration := ShortTest
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "medium":
			duration = MediumTest
		case "long":
			duration = LongTest
		}
	}

	result := runLongStabilityTest(modelPath, inputDataPath, duration, basePath)
	if result == nil {
		fmt.Println("测试失败")
		os.Exit(1)
	}

	// 保存结果
	resultData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("序列化结果失败: %v\n", err)
		os.Exit(1)
	}

	resultFile := filepath.Join(basePath, "results", "go_long_stability_enhanced_result.json")
	err = os.WriteFile(resultFile, resultData, 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n结果已保存到: %s\n", resultFile)
}
