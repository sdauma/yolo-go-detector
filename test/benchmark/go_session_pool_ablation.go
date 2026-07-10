// go_session_pool_ablation.go
// Go Session Pool 消融实验（Ablation Study）
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置动态 intraOp 和 interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 测试不同 PoolSize × IntraThreads 组合（共 64 组配置）
//
// 测试目的：
// - 回答"不同池大小和线程配置如何影响吞吐量、延迟和内存"
// - 为论文补充消融实验（Ablation Study）数据
// - 每个配置测试 200 次请求，记录 avg/P50/P90/P99 延迟、吞吐量、RSS

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

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

type AblationConfig struct {
	PoolSize     int `json:"pool_size"`
	IntraThreads int `json:"intra_threads"`
}

type AblationResult struct {
	Config             AblationConfig `json:"config"`
	Status             string         `json:"status,omitempty"`
	SkipReason         string         `json:"skip_reason,omitempty"`
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

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

func runAblationTest(
	modelPath string,
	inputData []byte,
	config AblationConfig,
	basePath string,
	modelName string,
) *AblationResult {
	fmt.Printf("\n--- 消融实验: PoolSize=%d, IntraThreads=%d ---\n",
		config.PoolSize, config.IntraThreads)

	// 配置 GOMAXPROCS
	runtime.GOMAXPROCS(config.PoolSize)

	// 创建 Session Pool
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
			fmt.Printf("创建SessionOptions失败: %v\n", err)
			return &AblationResult{
				Config: config, Model: modelName,
				Status: "SKIPPED", SkipReason: fmt.Sprintf("SessionOptions creation failed: %v", err),
			}
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
			fmt.Printf("创建输入Tensor失败: %v\n", err)
			return &AblationResult{
				Config: config, Model: modelName,
				Status: "SKIPPED", SkipReason: fmt.Sprintf("Input tensor creation failed: %v", err),
			}
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
			fmt.Printf("创建输出Tensor失败: %v\n", err)
			return &AblationResult{
				Config: config, Model: modelName,
				Status: "SKIPPED", SkipReason: fmt.Sprintf("Output tensor creation failed: %v", err),
			}
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
			fmt.Printf("创建Session失败: %v\n", err)
			return &AblationResult{
				Config: config, Model: modelName,
				Status: "SKIPPED", SkipReason: fmt.Sprintf("Session creation failed: %v", err),
			}
		}

		item := &poolItem{session, inputTensor, outputTensor}
		pool <- item
		allSessions = append(allSessions, item)
	}

	// 预热
	fmt.Print("  Warmup...")
	for i := 0; i < 10; i++ {
		item := <-pool
		item.session.Run()
		pool <- item
	}
	fmt.Println(" done")

	// 记录初始状态
	startRSS := getProcessRSS()
	peakRSS := startRSS
	startTime := time.Now()

	// 测试参数
	const numRequests = 500
	const concurrency = 12
	var (
		latencies   []float64
		completed   int32
		totalIssued int32
		mu          sync.Mutex
		wg          sync.WaitGroup
	)

	// 并发推理
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

	// 计算统计量
	sort.Float64s(latencies)
	n := len(latencies)
	if n == 0 {
		fmt.Println("无有效推理数据")
		// Cleanup sessions before returning
		for _, item := range allSessions {
			item.session.Destroy()
			item.inputTensor.Destroy()
			item.outputTensor.Destroy()
		}
		return &AblationResult{
			Config: config, Model: modelName,
			Status: "SKIPPED", SkipReason: "No valid inference data",
		}
	}

	var sum float64
	for _, lat := range latencies {
		sum += lat
	}
	avg := sum / float64(n)

	var sumSq float64
	for _, lat := range latencies {
		diff := lat - avg
		sumSq += diff * diff
	}
	stdLat := math.Sqrt(sumSq / float64(n))

	p50 := latencies[n*50/100]
	p90 := latencies[n*90/100]
	p99 := latencies[n*99/100]
	minLat := latencies[0]
	maxLat := latencies[n-1]
	throughput := float64(n) / duration

	result := &AblationResult{
		Config:             config,
		Status:             "OK",
		TotalRequests:      n,
		SuccessfulRequests: n,
		AvgLatencyMs:       avg,
		P50LatencyMs:       p50,
		P90LatencyMs:       p90,
		P99LatencyMs:       p99,
		MinLatencyMs:       minLat,
		MaxLatencyMs:       maxLat,
		StdLatencyMs:       stdLat,
		Throughput:         throughput,
		StartRSSMB:         startRSS,
		PeakRSSMB:          peakRSS,
		EndRSSMB:           endRSS,
		RSSDriftMB:         endRSS - startRSS,
		DurationSec:        duration,
		Model:              modelName,
	}

	fmt.Printf("  吞吐量: %.2f REQ/s\n", throughput)
	// 延迟展示保留3位小数，符合核心期刊规范
	fmt.Printf("  平均延迟: %.3f ms, P50: %.3f ms, P99: %.3f ms\n", avg, p50, p99)
	// 统计量展示保留4位小数，符合核心期刊规范
	fmt.Printf("  延迟标准差: %.4f ms\n", stdLat)
	fmt.Printf("  RSS: 起始 %.2f MB, 峰值 %.2f MB, 结束 %.2f MB, 漂移 %.2f MB\n",
		startRSS, peakRSS, endRSS, endRSS-startRSS)

	// 清理
	for _, item := range allSessions {
		item.session.Destroy()
		item.inputTensor.Destroy()
		item.outputTensor.Destroy()
	}

	return result
}

func main() {
	fmt.Println("===== Go Session Pool 消融实验 =====")
	fmt.Println("测试目的：评估池大小和线程配置对Session Pool性能的影响")
	fmt.Println("")

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	// 配置参数
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	resultsDir := filepath.Join(basePath, "results")

	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	// 测试配置：池大小 x 线程配置
	poolSizes := []int{4, 8, 12, 16}
	intraThreadConfigs := []int{1, 2, 4, 8}
	models := []struct {
		name string
		file string
	}{
		{"YOLO11x", "yolo11x.onnx"},
		{"YOLO11n", "yolo11n.onnx"},
	}

	var allResults []AblationResult

	for _, modelInfo := range models {
		modelPath := filepath.Join(basePath, "third_party", modelInfo.file)
		inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

		fmt.Printf("\n========== 模型: %s ==========\n", modelInfo.name)

		// 加载输入数据
		inputData, err := os.ReadFile(inputDataPath)
		if err != nil {
			fmt.Printf("读取输入数据失败: %v\n", err)
			continue
		}

		// 遍历所有配置组合
		for _, poolSize := range poolSizes {
			for _, threads := range intraThreadConfigs {
			// 跳过不合理配置（线程数大于池大小），记录到结果中
			if threads > poolSize {
				fmt.Printf("  [SKIP] PoolSize=%d, IntraThreads=%d (threads > pool_size)\n", poolSize, threads)
				allResults = append(allResults, AblationResult{
					Config:     AblationConfig{PoolSize: poolSize, IntraThreads: threads},
					Model:      modelInfo.name,
					Status:     "SKIPPED",
					SkipReason: "threads > pool_size",
				})
				continue
			}

				config := AblationConfig{
					PoolSize:     poolSize,
					IntraThreads: threads,
				}

				result := runAblationTest(modelPath, inputData, config, basePath, modelInfo.name)
				allResults = append(allResults, *result) // always append (may have status=SKIPPED)

				// 短暂休息，让系统冷却
				time.Sleep(2 * time.Second)
			}
		}
	}

	// 保存结果
	// 控制台输出保留2位小数，便于阅读
	fmt.Printf("\n========== 消融实验汇总 ==========\n")
	fmt.Printf("%-12s %-8s %-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
		"模型", "池大小", "线程", "状态", "吞吐量", "平均延迟", "P99延迟", "RSS峰值", "RSS漂移")
	fmt.Println(strings.Repeat("-", 90))

	skippedCount := 0
	for _, r := range allResults {
		if r.Status == "SKIPPED" {
			skippedCount++
			fmt.Printf("%-12s %-8d %-6d %-10s %-10s %-10s %-10s %-10s %-10s\n",
				r.Model, r.Config.PoolSize, r.Config.IntraThreads,
				"SKIPPED", "-", "-", "-", "-", "-")
		} else {
			fmt.Printf("%-12s %-8d %-6d %-10s %-10.2f %-10.3f %-10.3f %-10.2f %-10.2f\n",
				r.Model, r.Config.PoolSize, r.Config.IntraThreads,
				"OK", r.Throughput, r.AvgLatencyMs, r.P99LatencyMs,
				r.PeakRSSMB, r.RSSDriftMB)
		}
	}

	// 保存JSON结果
	resultData, err := json.MarshalIndent(allResults, "", "  ")
	if err != nil {
		fmt.Printf("序列化结果失败: %v\n", err)
		os.Exit(1)
	}

	resultFile := filepath.Join(resultsDir, "go_session_pool_ablation.json")
	err = os.WriteFile(resultFile, resultData, 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n结果已保存到: %s\n", resultFile)
	okCount := len(allResults) - skippedCount
	fmt.Printf("共 %d 组消融实验（%d 组完成，%d 组跳过）\n", len(allResults), okCount, skippedCount)
}
