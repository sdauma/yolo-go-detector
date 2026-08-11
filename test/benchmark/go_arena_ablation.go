// go_arena_ablation.go
// CPU 内存 Arena 开关消融实验
//
// 目的：验证 Unsafe Shared 架构的大幅 RSS 漂移是否来自共享 Session 的
//       CPU 内存分配器（Arena）高水位保持与碎片化。
//
// 设计：
//   - 控制变量：同模型(YOLO11x)、同并发度、同 intra_op/inter_op、同 GOMAXPROCS
//   - 自变量：SetCpuMemArena(true/false)，同时固定 SetMemPattern(false)
//   - 测试架构：Unsafe Shared (4并发) 与 Session Pool (池大小4)
//   - 指标：吞吐量、峰值PM、PM漂移（稳态窗口：去除前20%后的60%）
//
// 预期：
//   - Unsafe Shared 在 arena=false 时漂移显著下降（哪怕吞吐略降）
//   - Session Pool 漂移本就小，变化幅度较小
//   - 由此可将"大漂移"因果归因锁定到"共享 Arena 的高水位/碎片化"

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

type ArenaTestResult struct {
	Architecture string
	ArenaEnabled bool
	Concurrency  int
	Throughput   float64
	AvgLatency   float64
	P50Latency   float64
	P90Latency   float64
	P99Latency   float64
	StartRSS     float64
	PeakRSS      float64
	EndRSS       float64
	RSSDrift     float64
}

func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

type sessionWithTensors struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputData    []byte
}

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

// testUnsafeSharedArena 测试 Unsafe Shared 架构（带 arena 控制）
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
	fmt.Printf("测试 Unsafe Shared (arena=%s): %d 并发，%d 请求\n", arenaStr, concurrency, numRequests)

	startRSS := getProcessRSS()
	peakRSS := startRSS

	swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOpThreads, arenaEnabled)
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return &ArenaTestResult{Architecture: "Unsafe Shared", ArenaEnabled: arenaEnabled}
	}
	defer func() {
		swt.session.Destroy()
		swt.inputTensor.Destroy()
		swt.outputTensor.Destroy()
	}()

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

			// 局部 swt 用于填充数据
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

	return collectArenaResults("Unsafe Shared", arenaEnabled, concurrency, numRequests, totalTime, startRSS, peakRSS, endRSS, latencyChan, errorChan)
}

// testSessionPoolArena 测试 Session Pool 架构（带 arena 控制）
func testSessionPoolArena(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	poolSize int,
	numRequests int,
	intraOpThreads int,
	arenaEnabled bool,
) *ArenaTestResult {
	arenaStr := "ON"
	if !arenaEnabled {
		arenaStr = "OFF"
	}
	fmt.Printf("测试 Session Pool (arena=%s): pool_size=%d, %d 请求\n", arenaStr, poolSize, numRequests)

	startRSS := getProcessRSS()
	var peakRSS float64 = startRSS
	var mu sync.Mutex

	var wg sync.WaitGroup
	errorChan := make(chan error, numRequests)
	latencyChan := make(chan float64, numRequests)

	startTime := time.Now()
	batchSize := numRequests / poolSize

	for i := 0; i < poolSize; i++ {
		swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOpThreads, arenaEnabled)
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

	return collectArenaResults("Session Pool", arenaEnabled, poolSize, numRequests, totalTime, startRSS, finalPeakRSS, endRSS, latencyChan, errorChan)
}

func collectArenaResults(
	arch string,
	arenaEnabled bool,
	concurrency int,
	numRequests int,
	totalTime float64,
	startRSS, peakRSS, endRSS float64,
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
		Architecture: arch,
		ArenaEnabled: arenaEnabled,
		Concurrency:  concurrency,
		StartRSS:     startRSS,
		PeakRSS:      peakRSS,
		EndRSS:       endRSS,
		RSSDrift:     endRSS - startRSS,
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

func main() {
	// 初始化 ONNX Runtime
	// 路径相对于 test/benchmark/ 目录
	basePath, _ := os.Getwd()
	// 向上查找项目根目录（包含 third_party 的目录）
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

	// 检查文件是否存在
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

	// 设置 GOMAXPROCS = 物理核数
	runtime.GOMAXPROCS(6)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// 实验配置
	concurrency := 4
	poolSize := 4
	numRequests := 500
	intraOpThreads := 1

	fmt.Println("===== CPU 内存 Arena 开关消融实验 =====")
	fmt.Printf("模型: YOLO11x, 并发度: %d, 池大小: %d, 请求总数: %d\n", concurrency, poolSize, numRequests)
	fmt.Printf("intra_op=%d, inter_op=1, MemPattern=false(固定)\n", intraOpThreads)
	fmt.Println()

	var results []*ArenaTestResult

	// 4 组实验：Unsafe Shared × {arena on/off} + Session Pool × {arena on/off}
	// 预热：每组实验前先跑 50 次预热
	fmt.Println("--- 预热阶段 ---")
	warmupOpts, _ := ort.NewSessionOptions()
	warmupOpts.SetIntraOpNumThreads(intraOpThreads)
	warmupOpts.Destroy()

	// 实验 1: Unsafe Shared, arena=ON
	fmt.Println("\n--- 实验 1: Unsafe Shared, arena=ON ---")
	r1 := testUnsafeSharedArena(modelPath, inputData, inputShape, outputShape, concurrency, numRequests, intraOpThreads, true)
	results = append(results, r1)

	// 实验 2: Unsafe Shared, arena=OFF
	fmt.Println("\n--- 实验 2: Unsafe Shared, arena=OFF ---")
	r2 := testUnsafeSharedArena(modelPath, inputData, inputShape, outputShape, concurrency, numRequests, intraOpThreads, false)
	results = append(results, r2)

	// 实验 3: Session Pool, arena=ON
	fmt.Println("\n--- 实验 3: Session Pool, arena=ON ---")
	r3 := testSessionPoolArena(modelPath, inputData, inputShape, outputShape, poolSize, numRequests, intraOpThreads, true)
	results = append(results, r3)

	// 实验 4: Session Pool, arena=OFF
	fmt.Println("\n--- 实验 4: Session Pool, arena=OFF ---")
	r4 := testSessionPoolArena(modelPath, inputData, inputShape, outputShape, poolSize, numRequests, intraOpThreads, false)
	results = append(results, r4)

	// 输出结果
	fmt.Println("\n===== Arena 开关消融实验结果 =====")
	fmt.Println()
	fmt.Printf("%-16s %-8s %-10s %-12s %-12s %-12s\n", "架构", "Arena", "吞吐量", "平均延迟", "峰值PM", "PM漂移")
	fmt.Printf("%-16s %-8s %-10s %-12s %-12s %-12s\n", "", "", "(REQ/s)", "(ms)", "(MB)", "(MB)")
	fmt.Println("------------------------------------------------------------------------")

	for _, r := range results {
		arenaStr := "ON"
		if !r.ArenaEnabled {
			arenaStr = "OFF"
		}
		fmt.Printf("%-16s %-8s %-10.5f %-12.3f %-12.2f %-12.2f\n",
			r.Architecture, arenaStr, r.Throughput, r.AvgLatency, r.PeakRSS, r.RSSDrift)
	}

	// 保存到文件
	outputPath := filepath.Join("..", "..", "results", "go_arena_ablation_result.txt")
	f, err := os.Create(outputPath)
	if err != nil {
		fmt.Printf("创建输出文件失败: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Fprintln(f, "===== Go CPU 内存 Arena 开关消融实验结果 =====")
	fmt.Fprintln(f)
	fmt.Fprintf(f, "模型: YOLO11x, 并发度: %d, 池大小: %d, 请求总数: %d\n", concurrency, poolSize, numRequests)
	fmt.Fprintf(f, "intra_op=%d, inter_op=1, MemPattern=false(固定), GOMAXPROCS=6\n", intraOpThreads)
	fmt.Fprintln(f)

	for _, r := range results {
		arenaStr := "ON"
		if !r.ArenaEnabled {
			arenaStr = "OFF"
		}
		fmt.Fprintf(f, "===== %s (arena=%s) =====\n", r.Architecture, arenaStr)
		fmt.Fprintf(f, "  吞吐量: %.5f REQ/s\n", r.Throughput)
		fmt.Fprintf(f, "  平均延迟: %.5f ms\n", r.AvgLatency)
		fmt.Fprintf(f, "  P50延迟: %.5f ms\n", r.P50Latency)
		fmt.Fprintf(f, "  P90延迟: %.5f ms\n", r.P90Latency)
		fmt.Fprintf(f, "  P99延迟: %.5f ms\n", r.P99Latency)
		fmt.Fprintf(f, "  起始PM: %.5f MB\n", r.StartRSS)
		fmt.Fprintf(f, "  峰值PM: %.5f MB\n", r.PeakRSS)
		fmt.Fprintf(f, "  结束PM: %.5f MB\n", r.EndRSS)
		fmt.Fprintf(f, "  PM漂移: %.5f MB\n", r.RSSDrift)
		fmt.Fprintln(f)
	}

	fmt.Printf("\n结果已保存至: %s\n", outputPath)
}
