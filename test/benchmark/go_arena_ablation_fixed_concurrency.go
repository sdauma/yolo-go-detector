// go_arena_ablation_fixed_concurrency.go
// 固定并发Run()数、变池容量的 Arena 消融对照实验
//
// 目的：同时回答两个审稿关键问题：
//
//   P0-1（竞争假设）：Unsafe Shared + 关 Arena 的漂移(235MB)是否可接受？
//                     Session Pool 相对它的增量价值何在？
//                     —— 通过 Unsafe Shared(4并发) Arena OFF 与
//                        Session Pool(4并发, 各池大小) 的漂移对比回答。
//
//   P0-2（混淆变量）：原表8中池 8→12 漂移骤增，是池容量驱动还是
//                     并发 Run() 数驱动？原表8中 pool≤8 时 12 路协程受池
//                     容量限制(实际并发 Run()≤8)，pool≥12 时 12 路全并发。
//                     固定并发=4、仅变池容量即可分离两个变量。
//
// 设计：
//   - 固定 concurrency=4（并发 Run() 数），intra_op=1，500 次推理
//   - Session Pool 池大小扫描 {4,8,12,16}，每个池大小 Arena ON/OFF
//   - Unsafe Shared 4并发 Arena ON/OFF 作为对照基线
//   - 租用模式：goroutine 通过 channel Get/Put 复用 Session（与
//     engine.SessionPool 的 GetSession/PutSession 逻辑一致），
//     池大小>并发数时多余 Session 预创建后空闲（占用内存但不参与 Run()），
//     纯粹测试"池容量本身"对漂移的影响，与并发数解耦。
//
// 与原 go_arena_ablation.go（表9）的关系：
//   原实验为 4并发/4池/intra_op=1（固定分配模式，并发=池大小）。
//   本实验扩展为固定4并发、扫描池大小（租用模式，并发≠池大小）。
//   原实验的 4池数据可与本实验 pool=4 行交叉验证。
//
// 内存安全：pool=16 时预创建 16 个 Session（每个约 500MB），
//   创建过程中若峰值 PM 超过 14GB 则停止创建并记录实际池大小。

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
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

// ArenaTestResult 单组实验结果
type ArenaTestResult struct {
	Architecture   string  `json:"architecture"`
	ArenaEnabled   bool    `json:"arena_enabled"`
	Concurrency    int     `json:"concurrency"`     // 并发 Run() 数
	PoolSize       int     `json:"pool_size"`       // 配置的池容量
	ActualPoolSize int     `json:"actual_pool_size"` // 实际创建的 Session 数
	Throughput     float64 `json:"throughput"`
	AvgLatency     float64 `json:"avg_latency"`
	P50Latency     float64 `json:"p50_latency"`
	P90Latency     float64 `json:"p90_latency"`
	P99Latency     float64 `json:"p99_latency"`
	StartRSS       float64 `json:"start_rss"`        // Session 创建前
	PostCreateRSS  float64 `json:"post_create_rss"`  // Session 创建后（推理前）
	PeakRSS        float64 `json:"peak_rss"`         // 推理期间峰值
	EndRSS         float64 `json:"end_rss"`          // 推理结束
	RSSDrift       float64 `json:"rss_drift"`        // EndRSS - StartRSS（总漂移，含创建）
	InferenceDrift float64 `json:"inference_drift"`  // EndRSS - PostCreateRSS（仅推理阶段）
	ErrorCount     int     `json:"error_count"`
}

func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

type sessionWithTensors struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputData    []byte
}

// createSessionWithTensorsArena 创建带 Arena 控制的 Session
func createSessionWithTensorsArena(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	intraOpThreads int,
	arenaEnabled bool,
) (*sessionWithTensors, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建 SessionOptions 失败: %v", err)
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	// 固定关闭 MemPattern，避免额外预分配影响对比
	if err := opts.SetMemPattern(false); err != nil {
		return nil, fmt.Errorf("SetMemPattern 失败: %v", err)
	}

	// 自变量：arena 开关
	if err := opts.SetCpuMemArena(arenaEnabled); err != nil {
		return nil, fmt.Errorf("SetCpuMemArena 失败: %v", err)
	}

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入 Tensor 失败: %v", err)
	}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出 Tensor 失败: %v", err)
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
		return nil, fmt.Errorf("创建 Session 失败: %v", err)
	}

	return &sessionWithTensors{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputData:    inputData,
	}, nil
}

func fillInputData(inputTensor *ort.Tensor[float32], inputData []byte) {
	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

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

// testUnsafeSharedArena 测试 Unsafe Shared 架构（共享 1 个 Session，多 goroutine 并发 Run()）
// 与原 go_arena_ablation.go 中的实现一致，保证可比性。
func testUnsafeSharedArena(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	concurrency int,
	numRequests int,
	intraOpThreads int,
	arenaEnabled bool,
) *ArenaTestResult {
	arenaStr := "ON"
	if !arenaEnabled {
		arenaStr = "OFF"
	}
	fmt.Printf("  [Unsafe Shared] arena=%s, 并发=%d, 请求=%d\n", arenaStr, concurrency, numRequests)

	startRSS := getProcessRSS()

	swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOpThreads, arenaEnabled)
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return &ArenaTestResult{Architecture: "Unsafe Shared", ArenaEnabled: arenaEnabled, Concurrency: concurrency}
	}
	postCreateRSS := getProcessRSS()
	peakRSS := postCreateRSS

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// 每个 goroutine 创建独立的 tensor（共享 session）
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

			localSwt := &sessionWithTensors{
				session:      swt.session,
				inputTensor:  inputTensor,
				outputTensor: outputTensor,
				inputData:    inputData,
			}

			for j := 0; j < batchSize; j++ {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}

				start := time.Now()
				err := runInference(localSwt)
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

	// 销毁 session
	swt.session.Destroy()
	swt.inputTensor.Destroy()
	swt.outputTensor.Destroy()

	return collectResults("Unsafe Shared", arenaEnabled, concurrency, 1, 1,
		numRequests, totalTime, startRSS, postCreateRSS, peakRSS, endRSS, latencyChan, errorChan)
}

// testSessionPoolFixedConcurrency 固定并发 Run() 数、变池容量的 Session Pool 测试（租用模式）
//
// 与原 go_arena_ablation.go 的关键区别：
//   - 原版：poolSize 个 goroutine 各持有 1 个 session（并发=池大小，固定分配）
//   - 本版：concurrency 个 goroutine 通过 channel 租用 poolSize 个 session
//          （并发≠池大小，租用模式），池大小>并发数时多余 session 空闲
//
// 这样可分离"池容量"与"并发 Run() 数"两个变量。
func testSessionPoolFixedConcurrency(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	poolSize int,
	concurrency int,
	numRequests int,
	intraOpThreads int,
	arenaEnabled bool,
) *ArenaTestResult {
	arenaStr := "ON"
	if !arenaEnabled {
		arenaStr = "OFF"
	}
	fmt.Printf("  [Session Pool] arena=%s, 池大小=%d, 并发=%d, 请求=%d\n",
		arenaStr, poolSize, concurrency, numRequests)

	startRSS := getProcessRSS()

	// 预创建 poolSize 个 session，放入 channel（模拟 SessionPool 的预创建）
	pool := make(chan *sessionWithTensors, poolSize)
	actualPoolSize := 0
	for i := 0; i < poolSize; i++ {
		swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOpThreads, arenaEnabled)
		if err != nil {
			fmt.Printf("  创建 Session %d 失败: %v\n", i+1, err)
			continue
		}
		pool <- swt
		actualPoolSize++

		// 内存安全检查：峰值 PM 超过 14GB 则停止创建
		if getProcessRSS() > 14000 {
			fmt.Printf("  [WARN] 峰值PM超过14GB，停止创建（已创建 %d/%d）\n", actualPoolSize, poolSize)
			break
		}
	}

	if actualPoolSize == 0 {
		return &ArenaTestResult{
			Architecture: "Session Pool", ArenaEnabled: arenaEnabled,
			Concurrency: concurrency, PoolSize: poolSize, ActualPoolSize: 0,
		}
	}

	postCreateRSS := getProcessRSS()
	peakRSS := postCreateRSS
	var mu sync.Mutex

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	requestsPerWorker := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < requestsPerWorker; j++ {
				// Get session（租用）
				swt := <-pool

				// Run inference
				start := time.Now()
				err := runInference(swt)
				latency := float64(time.Since(start).Milliseconds())

				// Put session back（归还）
				pool <- swt

				// Track peak
				currentRSS := getProcessRSS()
				mu.Lock()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
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

	// 销毁所有 session
	close(pool)
	for swt := range pool {
		swt.session.Destroy()
		swt.inputTensor.Destroy()
		swt.outputTensor.Destroy()
	}

	return collectResults("Session Pool", arenaEnabled, concurrency, poolSize, actualPoolSize,
		numRequests, totalTime, startRSS, postCreateRSS, peakRSS, endRSS, latencyChan, errorChan)
}

func collectResults(
	arch string,
	arenaEnabled bool,
	concurrency int,
	poolSize int,
	actualPoolSize int,
	numRequests int,
	totalTime float64,
	startRSS, postCreateRSS, peakRSS, endRSS float64,
	latencyChan chan float64,
	errorChan chan error,
) *ArenaTestResult {
	var latencies []float64
	for lat := range latencyChan {
		latencies = append(latencies, lat)
	}
	errorCount := 0
	for range errorChan {
		errorCount++
	}

	result := &ArenaTestResult{
		Architecture:   arch,
		ArenaEnabled:   arenaEnabled,
		Concurrency:    concurrency,
		PoolSize:       poolSize,
		ActualPoolSize: actualPoolSize,
		StartRSS:       startRSS,
		PostCreateRSS:  postCreateRSS,
		PeakRSS:        peakRSS,
		EndRSS:         endRSS,
		RSSDrift:       endRSS - startRSS,
		InferenceDrift: endRSS - postCreateRSS,
		ErrorCount:     errorCount,
	}

	if len(latencies) > 0 {
		sort.Float64s(latencies)
		sum := 0.0
		for _, l := range latencies {
			sum += l
		}
		result.AvgLatency = sum / float64(len(latencies))
		result.P50Latency = calculatePercentile(latencies, 0.50)
		result.P90Latency = calculatePercentile(latencies, 0.90)
		result.P99Latency = calculatePercentile(latencies, 0.99)
	}

	if totalTime > 0 {
		result.Throughput = float64(len(latencies)) / (totalTime / 1000.0)
	}

	return result
}

// cleanupBetweenTests 组间清理：GC + 等待，减少前一组残留内存干扰
func cleanupBetweenTests() {
	runtime.GC()
	time.Sleep(3 * time.Second)
}

func main() {
	// 定位项目根目录
	basePath, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(basePath, "third_party")); err == nil {
			break
		}
		parent := filepath.Dir(basePath)
		if parent == basePath {
			break
		}
		basePath = parent
	}
	fmt.Printf("项目根目录: %s\n", basePath)

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	fmt.Printf("库路径: %s\n", libPath)
	fmt.Printf("模型路径: %s\n", modelPath)
	fmt.Printf("输入数据路径: %s\n", inputDataPath)

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		fmt.Printf("错误: ONNX Runtime 库不存在: %s\n", libPath)
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
	fmt.Printf("输入数据大小: %d bytes\n", len(inputData))

	runtime.GOMAXPROCS(6)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// 实验配置（与原 arena 消融表9保持一致，保证可比）
	concurrency := 4        // 固定并发 Run() 数
	numRequests := 500      // 每组推理次数
	intraOpThreads := 1     // 与原实验一致

	fmt.Println()
	fmt.Println("===== 固定并发 Arena 消融对照实验 =====")
	fmt.Printf("模型: YOLO11x, 固定并发=%d, intra_op=%d, 每组%d次推理\n", concurrency, intraOpThreads, numRequests)
	fmt.Printf("池大小扫描: 4/6/8/12/16, 每点 Arena ON/OFF\n")
	fmt.Println("目的: 分离池容量与并发数对漂移的影响；对比 Unsafe Shared+关Arena vs Session Pool")
	fmt.Println()

	var results []*ArenaTestResult

	// ===== 对照组：Unsafe Shared (4并发) × Arena {ON, OFF} =====
	fmt.Println("--- [1/12] Unsafe Shared, arena=ON ---")
	r := testUnsafeSharedArena(modelPath, inputData, inputShape, outputShape, concurrency, numRequests, intraOpThreads, true)
	results = append(results, r)
	cleanupBetweenTests()

	fmt.Println("--- [2/12] Unsafe Shared, arena=OFF ---")
	r = testUnsafeSharedArena(modelPath, inputData, inputShape, outputShape, concurrency, numRequests, intraOpThreads, false)
	results = append(results, r)
	cleanupBetweenTests()

	// ===== 实验组：Session Pool (并发=4固定) × 池大小{4,6,8,12,16} × Arena{ON,OFF} =====
	poolSizes := []int{4, 6, 8, 12, 16}
	expIdx := 3
	for _, ps := range poolSizes {
		for _, arena := range []bool{true, false} {
			arenaStr := "ON"
			if !arena {
				arenaStr = "OFF"
			}
			fmt.Printf("--- [%d/12] Session Pool, 池大小=%d, arena=%s ---\n", expIdx, ps, arenaStr)
			r = testSessionPoolFixedConcurrency(modelPath, inputData, inputShape, outputShape,
				ps, concurrency, numRequests, intraOpThreads, arena)
			results = append(results, r)
			cleanupBetweenTests()
			expIdx++
		}
	}

	// ===== 输出结果表格 =====
	fmt.Println()
	fmt.Println("===== 固定并发 Arena 消融对照实验结果 =====")
	fmt.Println()
	fmt.Printf("%-16s %-6s %-6s %-6s %-10s %-10s %-12s %-12s %-12s\n",
		"架构", "Arena", "并发", "池大小", "吞吐量", "平均延迟", "峰值PM", "PM漂移", "推理漂移")
	fmt.Printf("%-16s %-6s %-6s %-6s %-10s %-10s %-12s %-12s %-12s\n",
		"", "", "", "", "(REQ/s)", "(ms)", "(MB)", "(MB)", "(MB)")
	fmt.Println("--------------------------------------------------------------------------------------------------------")

	for _, r := range results {
		arenaStr := "ON"
		if !r.ArenaEnabled {
			arenaStr = "OFF"
		}
		poolStr := "-"
		if r.Architecture == "Session Pool" {
			poolStr = fmt.Sprintf("%d(%d)", r.PoolSize, r.ActualPoolSize)
		}
		fmt.Printf("%-16s %-6s %-6d %-6s %-10.5f %-10.3f %-12.2f %-12.2f %-12.2f\n",
			r.Architecture, arenaStr, r.Concurrency, poolStr,
			r.Throughput, r.AvgLatency, r.PeakRSS, r.RSSDrift, r.InferenceDrift)
	}

	// ===== 保存 TXT 结果 =====
	outputPath := filepath.Join("..", "..", "results", "go_arena_ablation_fixed_concurrency_result.txt")
	f, err := os.Create(outputPath)
	if err != nil {
		fmt.Printf("创建输出文件失败: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Fprintln(f, "===== Go 固定并发 Arena 消融对照实验结果 =====")
	fmt.Fprintln(f)
	fmt.Fprintf(f, "模型: YOLO11x, 固定并发=%d, intra_op=%d, inter_op=1, MemPattern=false(固定), GOMAXPROCS=6\n", concurrency, intraOpThreads)
	fmt.Fprintf(f, "每组%d次推理, 池大小扫描: 4/6/8/12/16\n", numRequests)
	fmt.Fprintf(f, "租用模式: goroutine通过channel Get/Put复用Session, 池大小>并发数时多余Session空闲\n")
	fmt.Fprintln(f)
	fmt.Fprintf(f, "%-16s %-6s %-6s %-8s %-10s %-10s %-12s %-12s %-14s %-12s %-12s %-12s\n",
		"架构", "Arena", "并发", "池大小", "吞吐量", "平均延迟", "P50延迟", "P99延迟",
		"峰值PM(MB)", "起始PM(MB)", "PM漂移(MB)", "推理漂移(MB)")
	fmt.Fprintln(f, "------------------------------------------------------------------------------------------------------------------------")

	for _, r := range results {
		arenaStr := "ON"
		if !r.ArenaEnabled {
			arenaStr = "OFF"
		}
		poolStr := "1"
		if r.Architecture == "Session Pool" {
			poolStr = fmt.Sprintf("%d(%d)", r.PoolSize, r.ActualPoolSize)
		}
		fmt.Fprintf(f, "%-16s %-6s %-6d %-8s %-10.5f %-10.3f %-12.3f %-12.3f %-14.2f %-12.2f %-12.2f %-12.2f\n",
			r.Architecture, arenaStr, r.Concurrency, poolStr,
			r.Throughput, r.AvgLatency, r.P50Latency, r.P99Latency,
			r.PeakRSS, r.StartRSS, r.RSSDrift, r.InferenceDrift)
	}

	fmt.Fprintln(f)
	fmt.Fprintln(f, "说明:")
	fmt.Fprintln(f, "  PM漂移 = 结束PM - 起始PM(创建前), 含Session创建+推理两阶段")
	fmt.Fprintln(f, "  推理漂移 = 结束PM - 创建后PM, 仅推理阶段")
	fmt.Fprintln(f, "  池大小列: 配置值(实际创建数), 实际创建数<配置值时表示触达14GB内存安全阈值")
	fmt.Fprintln(f, "  并发=4固定, 池大小>4时多余Session空闲不参与Run(), 纯粹测试池容量对漂移的影响")

	fmt.Printf("\nTXT 结果已保存至: %s\n", outputPath)

	// ===== 保存 JSON 结果 =====
	jsonPath := filepath.Join("..", "..", "results", "go_arena_ablation_fixed_concurrency_result.json")
	jf, err := os.Create(jsonPath)
	if err != nil {
		fmt.Printf("创建JSON输出文件失败: %v\n", err)
		os.Exit(1)
	}
	defer jf.Close()

	jsonData, _ := json.MarshalIndent(struct {
		Config   map[string]interface{} `json:"config"`
		Results  []*ArenaTestResult     `json:"results"`
	}{
		Config: map[string]interface{}{
			"model":             "YOLO11x",
			"concurrency":       concurrency,
			"intra_op":          intraOpThreads,
			"inter_op":          1,
			"mem_pattern":       false,
			"gomaxprocs":        6,
			"num_requests":      numRequests,
			"pool_sizes":        poolSizes,
			"mode":              "rental (Get/Put via channel)",
			"memory_metric":     "PrivateMemorySize64 (PM)",
		},
		Results: results,
	}, "", "  ")
	fmt.Fprintln(jf, string(jsonData))

	fmt.Printf("JSON 结果已保存至: %s\n", jsonPath)
}
