// go_supplementary_ablation.go
// 补测程序：补全论文表8缺失的 4×8 配置 + 表7缺失的 Go intra_op=12
//
// 补测1（表8 4×8）：
//   原表8中 pool=4/intra=8 因 "threads > pool_size" 被跳过，但该跳过理由不成立——
//   intra_op_num_threads 是每个 Session 内部的算子并行线程数，与池大小无关。
//   8×8（64线程）已测，4×8（32线程）不测，逻辑不一致。
//   本补测用与原 go_session_pool_ablation.go 完全相同的配置补全 4×8。
//
// 补测2（表7 Go intra_op=12）：
//   原表7只有 Go intra_op=4/8 两个同线程对比点，补 intra_op=12 后变为 3 个点
//   (4/8/12)，直接证明"在 Python 使用的 12 线程下 Go 也差不多"。
//
// 输出：
//   results/go_ablation_4x8_supplement.json  （表8 4×8 数据，YOLO11x + YOLO11n）
//   results/go_thread_config_12_supplement.txt （表7 Go intra_op=12 数据）

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

// ========== 补测1：表8 4×8（复用 go_session_pool_ablation.go 逻辑）==========

type AblationConfig struct {
	PoolSize     int `json:"pool_size"`
	IntraThreads int `json:"intra_threads"`
}

type AblationResult struct {
	Config             AblationConfig `json:"config"`
	Status             string         `json:"status"`
	TotalRequests      int            `json:"total_requests"`
	SuccessfulRequests int            `json:"successful_requests"`
	AvgLatencyMs       float64        `json:"avg_latency_ms"`
	P50LatencyMs       float64        `json:"p50_latency_ms"`
	P90LatencyMs       float64        `json:"p90_latency_ms"`
	P99LatencyMs       float64        `json:"p99_latency_ms"`
	MinLatencyMs       float64        `json:"min_latency_ms"`
	MaxLatencyMs       float64        `json:"max_latency_ms"`
	StdLatencyMs       float64        `json:"std_latency_ms"`
	Throughput         float64        `json:"throughput_reqs"`
	StartRSSMB         float64        `json:"start_rss_mb"`
	PeakRSSMB          float64        `json:"peak_rss_mb"`
	EndRSSMB           float64        `json:"end_rss_mb"`
	RSSDriftMB         float64        `json:"rss_drift_mb"`
	DurationSec        float64        `json:"duration_sec"`
	Model              string         `json:"model"`
}

func runAblationTest(modelPath string, inputData []byte, config AblationConfig, modelName string) *AblationResult {
	fmt.Printf("\n--- 补测消融: PoolSize=%d, IntraThreads=%d, Model=%s ---\n",
		config.PoolSize, config.IntraThreads, modelName)

	runtime.GOMAXPROCS(config.PoolSize)

	type poolItem struct {
		session      *ort.AdvancedSession
		inputTensor  *ort.Tensor[float32]
		outputTensor *ort.Tensor[float32]
	}

	pool := make(chan *poolItem, config.PoolSize)
	var allSessions []*poolItem

	for i := 0; i < config.PoolSize; i++ {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			return &AblationResult{Config: config, Model: modelName, Status: "SKIPPED"}
		}
		opts.SetIntraOpNumThreads(config.IntraThreads)
		opts.SetInterOpNumThreads(1)
		opts.SetExecutionMode(0)
		opts.SetGraphOptimizationLevel(3)

		inputShape := []int64{1, 3, 640, 640}
		outputShape := []int64{1, 84, 8400}

		inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
		if err != nil {
			opts.Destroy()
			return &AblationResult{Config: config, Model: modelName, Status: "SKIPPED"}
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
			opts.Destroy()
			return &AblationResult{Config: config, Model: modelName, Status: "SKIPPED"}
		}

		session, err := ort.NewAdvancedSession(
			modelPath,
			[]string{"images"}, []string{"output0"},
			[]ort.Value{inputTensor}, []ort.Value{outputTensor},
			opts,
		)
		opts.Destroy()
		if err != nil {
			inputTensor.Destroy()
			outputTensor.Destroy()
			return &AblationResult{Config: config, Model: modelName, Status: "SKIPPED"}
		}

		item := &poolItem{session, inputTensor, outputTensor}
		pool <- item
		allSessions = append(allSessions, item)
	}

	// 预热（与原程序一致：10次）
	fmt.Print("  Warmup...")
	for i := 0; i < 10; i++ {
		item := <-pool
		item.session.Run()
		pool <- item
	}
	fmt.Println(" done")

	startRSS := getProcessRSS()
	peakRSS := startRSS
	startTime := time.Now()

	const numRequests = 500
	const concurrency = 12
	var (
		latencies   []float64
		completed   int32
		totalIssued int32
		mu          sync.Mutex
		wg          sync.WaitGroup
	)

	fmt.Print("  Progress: ")
	for c := 0; c < concurrency; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				idx := int(atomic.AddInt32(&totalIssued, 1))
				if idx > numRequests {
					break
				}
				item := <-pool

				inferStart := time.Now()
				err := item.session.Run()
				lat := float64(time.Since(inferStart).Milliseconds())

				pool <- item

				if err != nil {
					continue
				}

				mu.Lock()
				latencies = append(latencies, lat)
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
				done := int(atomic.AddInt32(&completed, 1))
				if done%100 == 0 {
					fmt.Printf("%d ", done)
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	endRSS := getProcessRSS()
	duration := time.Since(startTime).Seconds()

	sort.Float64s(latencies)
	n := len(latencies)
	if n == 0 {
		for _, item := range allSessions {
			item.session.Destroy()
			item.inputTensor.Destroy()
			item.outputTensor.Destroy()
		}
		return &AblationResult{Config: config, Model: modelName, Status: "SKIPPED"}
	}

	var sum float64
	for _, lat := range latencies {
		sum += lat
	}
	avg := sum / float64(n)

	var sumSq float64
	for _, lat := range latencies {
		sumSq += (lat - avg) * (lat - avg)
	}
	stdLat := math.Sqrt(sumSq / float64(n))

	result := &AblationResult{
		Config:             config,
		Status:             "OK",
		TotalRequests:      n,
		SuccessfulRequests: n,
		AvgLatencyMs:       avg,
		P50LatencyMs:       latencies[n*50/100],
		P90LatencyMs:       latencies[n*90/100],
		P99LatencyMs:       latencies[n*99/100],
		MinLatencyMs:       latencies[0],
		MaxLatencyMs:       latencies[n-1],
		StdLatencyMs:       stdLat,
		Throughput:         float64(n) / duration,
		StartRSSMB:         startRSS,
		PeakRSSMB:          peakRSS,
		EndRSSMB:           endRSS,
		RSSDriftMB:         endRSS - startRSS,
		DurationSec:        duration,
		Model:              modelName,
	}

	fmt.Printf("\n  吞吐量: %.5f REQ/s, 平均延迟: %.3f ms, P99: %.3f ms\n",
		result.Throughput, result.AvgLatencyMs, result.P99LatencyMs)
	fmt.Printf("  RSS: 起始 %.2f, 峰值 %.2f, 结束 %.2f, 漂移 %.2f MB\n",
		startRSS, peakRSS, endRSS, result.RSSDriftMB)

	for _, item := range allSessions {
		item.session.Destroy()
		item.inputTensor.Destroy()
		item.outputTensor.Destroy()
	}

	return result
}

// ========== 补测2：表7 Go intra_op=12（复用 thread_config_benchmark.go 逻辑）==========

func loadInputDataFromFile(data []float32, filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()
	buffer := make([]byte, len(data)*4)
	_, err = file.Read(buffer)
	if err != nil {
		return err
	}
	for i := 0; i < len(data); i++ {
		offset := i * 4
		u32 := binary.LittleEndian.Uint32(buffer[offset : offset+4])
		data[i] = *(*float32)(unsafe.Pointer(&u32))
	}
	return nil
}

func runThreadConfigTest(modelPath string, inputDataPath string, numThreads int) (float64, float64, float64) {
	fmt.Printf("\n--- 补测线程配置: intra_op=%d ---\n", numThreads)

	var allAvgLatencies []float64
	var allPeakRSS []float64
	var allStableRSS []float64

	testCount := 5
	for testIdx := 1; testIdx <= testCount; testIdx++ {
		fmt.Printf("  独立测试 %d/%d...", testIdx, testCount)

		opts, err := ort.NewSessionOptions()
		if err != nil {
			fmt.Printf(" 失败: %v\n", err)
			continue
		}
		opts.SetIntraOpNumThreads(numThreads)
		opts.SetInterOpNumThreads(1)

		inputShape := ort.NewShape(1, 3, 640, 640)
		inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
		if err != nil {
			opts.Destroy()
			fmt.Printf(" 失败\n")
			continue
		}

		inputData := inputTensor.GetData()
		if err := loadInputDataFromFile(inputData, inputDataPath); err != nil {
			inputTensor.Destroy()
			opts.Destroy()
			fmt.Printf(" 失败\n")
			continue
		}

		outputShape := ort.NewShape(1, 84, 8400)
		outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			inputTensor.Destroy()
			opts.Destroy()
			fmt.Printf(" 失败\n")
			continue
		}

		session, err := ort.NewAdvancedSession(modelPath,
			[]string{"images"}, []string{"output0"},
			[]ort.Value{inputTensor}, []ort.Value{outputTensor}, opts)
		if err != nil {
			inputTensor.Destroy()
			outputTensor.Destroy()
			opts.Destroy()
			fmt.Printf(" 失败\n")
			continue
		}

		startRSS := getProcessRSS()

		// Warmup 10次
		for i := 0; i < 10; i++ {
			session.Run()
		}

		// Benchmark 100次
		runs := 100
		times := make([]float64, runs)
		peakRSS := startRSS
		var sum float64
		for i := 0; i < runs; i++ {
			t0 := time.Now()
			session.Run()
			dt := float64(time.Since(t0).Milliseconds())
			times[i] = dt
			sum += dt
			if i%10 == 0 {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
			}
		}
		stableRSS := getProcessRSS()
		avgLat := sum / float64(runs)

		allAvgLatencies = append(allAvgLatencies, avgLat)
		allPeakRSS = append(allPeakRSS, peakRSS)
		allStableRSS = append(allStableRSS, stableRSS)

		fmt.Printf(" 平均延迟=%.3f ms\n", avgLat)

		session.Destroy()
		inputTensor.Destroy()
		outputTensor.Destroy()
		opts.Destroy()
	}

	if len(allAvgLatencies) == 0 {
		return 0, 0, 0
	}

	var totalLat, totalPeak, totalStable float64
	for i := range allAvgLatencies {
		totalLat += allAvgLatencies[i]
		totalPeak += allPeakRSS[i]
		totalStable += allStableRSS[i]
	}
	cnt := float64(len(allAvgLatencies))

	// 计算标准差
	avgLat := totalLat / cnt
	var sumSq float64
	for _, l := range allAvgLatencies {
		sumSq += (l - avgLat) * (l - avgLat)
	}
	stdLat := math.Sqrt(sumSq / cnt)

	fmt.Printf("  最终: 平均延迟=%.5f ms, 标准差=%.5f ms, Peak PM=%.2f MB\n",
		avgLat, stdLat, totalPeak/cnt)

	return avgLat, stdLat, totalPeak / cnt
}

// ========== main ==========

func main() {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	resultsDir := filepath.Join(basePath, "results")

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("初始化 ONNX Runtime 失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		os.Exit(1)
	}

	runtime.GOMAXPROCS(6)

	// ================================================================
	// 补测1：表8 4×8（YOLO11x + YOLO11n）
	// ================================================================
	fmt.Println()
	fmt.Println("===== 补测1: 表8 4×8 消融实验 =====")
	fmt.Println("（原表8因 threads>pool_size 跳过，补全以闭合实验设计）")

	config4x8 := AblationConfig{PoolSize: 4, IntraThreads: 8}
	var ablationResults []AblationResult

	// YOLO11x
	modelPath11x := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	fmt.Printf("\n>>> YOLO11x 4×8 <<<")
	r := runAblationTest(modelPath11x, inputData, config4x8, "YOLO11x")
	ablationResults = append(ablationResults, *r)
	time.Sleep(3 * time.Second)

	// YOLO11n
	modelPath11n := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	fmt.Printf("\n>>> YOLO11n 4×8 <<<")
	r = runAblationTest(modelPath11n, inputData, config4x8, "YOLO11n")
	ablationResults = append(ablationResults, *r)

	// 保存 JSON
	ablationJSON, _ := json.MarshalIndent(ablationResults, "", "  ")
	ablationPath := filepath.Join(resultsDir, "go_ablation_4x8_supplement.json")
	os.WriteFile(ablationPath, ablationJSON, 0644)
	fmt.Printf("\n表8 4×8 补测结果已保存: %s\n", ablationPath)

	// 打印汇总
	fmt.Printf("\n%-12s %-8s %-6s %-10s %-10s %-10s %-10s %-10s\n",
		"模型", "池大小", "线程", "吞吐量", "平均延迟", "P99延迟", "PM峰值", "PM漂移")
	fmt.Println(strings.Repeat("-", 80))
	for _, r := range ablationResults {
		if r.Status == "OK" {
			fmt.Printf("%-12s %-8d %-6d %-10.5f %-10.3f %-10.3f %-10.2f %-10.2f\n",
				r.Model, r.Config.PoolSize, r.Config.IntraThreads,
				r.Throughput, r.AvgLatencyMs, r.P99LatencyMs, r.PeakRSSMB, r.RSSDriftMB)
		}
	}

	// ================================================================
	// 补测2：表7 Go intra_op=12
	// ================================================================
	fmt.Println()
	fmt.Println("===== 补测2: 表7 Go intra_op=12 线程配置 =====")
	fmt.Println("（原表7只有 intra_op=4/8，补全 intra_op=12 使同线程对比点变为3个）")

	avgLat12, stdLat12, peakRSS12 := runThreadConfigTest(modelPath11x, inputDataPath, 12)

	// 保存结果
	threadPath := filepath.Join(resultsDir, "go_thread_config_12_supplement.txt")
	tf, err := os.Create(threadPath)
	if err != nil {
		fmt.Printf("创建输出文件失败: %v\n", err)
	} else {
		fmt.Fprintf(tf, "===== Go 线程配置补测: intra_op_num_threads=12（5次运行平均值）=====\n")
		fmt.Fprintf(tf, "模型: YOLO11x\n")
		fmt.Fprintf(tf, "平均延迟: %.5f ms\n", avgLat12)
		fmt.Fprintf(tf, "标准差: %.5f ms\n", stdLat12)
		fmt.Fprintf(tf, "FPS: %.2f\n", 1000.0/avgLat12)
		fmt.Fprintf(tf, "Peak PM: %.2f MB\n", peakRSS12)
		fmt.Fprintf(tf, "\n注: 与 thread_config_benchmark.go 完全相同的测试方法（5次独立测试×100次推理）\n")
		fmt.Fprintf(tf, "用途: 补全表7同线程配置对比（Go intra_op=12 vs Python intra_op=12=588.12ms）\n")
		tf.Close()
		fmt.Printf("表7 Go intra_op=12 补测结果已保存: %s\n", threadPath)
	}

	fmt.Println("\n===== 补测全部完成 =====")
}
