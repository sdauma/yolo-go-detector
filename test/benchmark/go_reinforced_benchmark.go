// go_reinforced_benchmark.go
// Go 强化测试（YOLO11x）- 10轮×200次推理
//
// 技术说明：
// - 使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
//   自动启用 I/O Binding，但不接受 SessionOptions 参数
// - 线程配置由 ONNX Runtime 默认行为决定（intra_op_num_threads 默认等于 CPU 核数）
// - 代码中创建了 SessionOptions 并设置了 intraOp=12，但由于 NewSession 不接受 opts，
//   这些设置实际上不生效。保留 opts 创建代码仅用于记录意图
//
// 测试目的：
// - 执行 10 轮×200 次推理，每轮前 warmup 20 次
// - 记录详细性能指标（avg/p50/p90/p95 延迟、RSS 内存），用于 t-test 分析
// - 确保数据稳定性和可重复性

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

// Rand 简单的随机数生成器，用于生成固定种子的随机数
type Rand struct {
	seed uint64
}

// Float32 生成 [0, 1) 范围的随机浮点数
func (r *Rand) Float32() float32 {
	r.seed = r.seed*1103515245 + 12345
	return float32((r.seed/65536)%32768) / 32768.0
}

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

// BenchmarkResult 单次测试结果
type BenchmarkResult struct {
	AvgLatency float64
	StdLatency float64
	P50Latency float64
	P90Latency float64
	P95Latency float64
	MinLatency float64
	MaxLatency float64
	StartRSS   float64
	PeakRSS    float64
	StableRSS  float64
	GoHeap     float64
	Times      []float64
}

// runBenchmark 执行一次基准测试
func runBenchmark() (*BenchmarkResult, error) {
	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("获取工作目录失败: %v", err)
	}

	// 构建项目根路径
	basePath := filepath.Dir(filepath.Dir(wd))

	// 设置模型和库路径
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")

	// 检查文件是否存在
	if !fileExists(modelPath) {
		return nil, fmt.Errorf("模型文件不存在: %s", modelPath)
	}
	if !fileExists(libPath) {
		return nil, fmt.Errorf("库文件不存在: %s", libPath)
	}

	// 初始化ORT
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 注意：NewSession 接口不接受 SessionOptions 参数，
	// 线程配置由 ONNX Runtime 默认行为决定（intra_op_num_threads 默认等于 CPU 核数）。
	// 如需自定义线程数，请使用 NewAdvancedSession 接口。

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入张量失败: %v", err)
	}
	defer inputTensor.Destroy()

	// 从预生成的二进制文件加载输入数据
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	inputDataFile, err := os.ReadFile(inputDataPath)
	if err != nil {
		return nil, fmt.Errorf("读取输入数据文件失败: %v", err)
	}

	// 转换为 float32 并填充到张量
	inputData := inputTensor.GetData()
	expectedSize := 1 * 3 * 640 * 640 * 4 // float32 = 4 bytes
	if len(inputDataFile) != expectedSize {
		return nil, fmt.Errorf("输入数据文件大小不匹配: 期望 %d 字节，实际 %d 字节", expectedSize, len(inputDataFile))
	}

	// 将字节数据转换为 float32 并复制到张量
	for i := 0; i < len(inputData); i += 4 {
		bits := binary.LittleEndian.Uint32(inputDataFile[i : i+4])
		value := math.Float32frombits(bits)
		inputData[i/4] = value
	}

	// 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输出张量失败: %v", err)
	}
	defer outputTensor.Destroy()

	// 创建会话
	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("创建会话失败: %v", err)
	}
	defer session.Destroy()

	// 内存采样点 1：Session 创建后、warmup 前（Start PM）
	startRSS := getProcessRSS()

	// Warmup
	for i := 0; i < 20; i++ { // 20次warmup
		if err := session.Run(); err != nil {
			return nil, fmt.Errorf("Warmup 运行失败: %v", err)
		}
	}

	// Benchmark
	runs := 200 // 每轮200次推理
	var sum float64
	times := make([]float64, runs)
	peakRSS := startRSS

	for i := 0; i < runs; i++ {
		t0 := time.Now()
		if err := session.Run(); err != nil {
			return nil, fmt.Errorf("运行失败: %v", err)
		}
		dt := time.Since(t0).Seconds() * 1000.0
		sum += dt
		times[i] = dt

		// 每10次推理采样一次内存，记录峰值
		if i%10 == 0 {
			currentRSS := getProcessRSS()
			if currentRSS > peakRSS {
				peakRSS = currentRSS
			}
		}
	}

	// 内存采样点 3：Benchmark 后稳定值
	stableRSS := getProcessRSS()

	// 计算结果
	sort.Float64s(times)
	avg_latency := sum / float64(runs)
	min_latency := times[0]
	max_latency := times[runs-1]
	p50_latency := times[int(float64(runs)*0.5)]
	p90_latency := times[int(float64(runs)*0.9)]
	p95_latency := times[int(float64(runs)*0.95)]

	// 计算标准差
	var variance float64
	for _, t := range times {
		variance += math.Pow(t-avg_latency, 2)
	}
	std_latency := math.Sqrt(variance / float64(runs))

	// 获取 Go heap 内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return &BenchmarkResult{
		AvgLatency: avg_latency,
		StdLatency: std_latency,
		P50Latency: p50_latency,
		P90Latency: p90_latency,
		P95Latency: p95_latency,
		MinLatency: min_latency,
		MaxLatency: max_latency,
		StartRSS:   startRSS,
		PeakRSS:    peakRSS,
		StableRSS:  stableRSS,
		GoHeap:     float64(m.Alloc) / 1024 / 1024,
		Times:      times,
	}, nil
}

func main() {
	fmt.Println("===== Go 强化测试（10轮运行） =====")

	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取工作目录失败: %v\n", err)
		return
	}

	// 构建项目根路径
	basePath := filepath.Dir(filepath.Dir(wd))

	// 运行10次测试
	numRuns := 10
	results := make([]*BenchmarkResult, numRuns)

	for i := 0; i < numRuns; i++ {
		fmt.Printf("\n===== 第 %d 轮测试 =====\n", i+1)
		result, err := runBenchmark()
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		results[i] = result

		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Printf("平均延迟: %.5f ms\n", result.AvgLatency)
		fmt.Printf("标准差: %.5f ms\n", result.StdLatency)
		fmt.Printf("P50延迟: %.5f ms\n", result.P50Latency)
		fmt.Printf("P90延迟: %.5f ms\n", result.P90Latency)
		fmt.Printf("P95延迟: %.5f ms\n", result.P95Latency)
		fmt.Printf("最小延迟: %.5f ms\n", result.MinLatency)
		fmt.Printf("最大延迟: %.5f ms\n", result.MaxLatency)
		fmt.Printf("Start PM: %.2f MB\n", result.StartRSS)
		fmt.Printf("Peak PM: %.2f MB\n", result.PeakRSS)
		fmt.Printf("Stable PM: %.2f MB\n", result.StableRSS)
		fmt.Printf("PM Drift: %.2f MB\n", result.StableRSS-result.StartRSS)
		fmt.Printf("Go Heap: %.2f MB\n", result.GoHeap)
	}

	// 计算平均值
	var avgLatency, stdLatency, p50Latency, p90Latency, p95Latency float64
	var minLatency, maxLatency float64
	var startRSS, peakRSS, stableRSS, goHeap float64

	for _, r := range results {
		avgLatency += r.AvgLatency
		stdLatency += r.StdLatency
		p50Latency += r.P50Latency
		p90Latency += r.P90Latency
		p95Latency += r.P95Latency
		minLatency += r.MinLatency
		maxLatency += r.MaxLatency
		startRSS += r.StartRSS
		peakRSS += r.PeakRSS
		stableRSS += r.StableRSS
		goHeap += r.GoHeap
	}

	avgLatency /= float64(numRuns)
	stdLatency /= float64(numRuns)
	p50Latency /= float64(numRuns)
	p90Latency /= float64(numRuns)
	p95Latency /= float64(numRuns)
	minLatency /= float64(numRuns)
	maxLatency /= float64(numRuns)
	startRSS /= float64(numRuns)
	peakRSS /= float64(numRuns)
	stableRSS /= float64(numRuns)
	goHeap /= float64(numRuns)

	// 计算吞吐量
	inferencesPerRun := 200
	totalInferences := numRuns * inferencesPerRun // 10轮 × 200次 = 2000次
	totalTimeSeconds := avgLatency * float64(numRuns) * float64(inferencesPerRun) / 1000.0
	throughput := float64(totalInferences) / totalTimeSeconds

	fmt.Printf("\n===== 10轮测试平均值 =====\n")
	fmt.Printf("平均延迟: %.3f ms\n", avgLatency)
	fmt.Printf("标准差: %.3f ms\n", stdLatency)
	fmt.Printf("P50延迟: %.3f ms\n", p50Latency)
	fmt.Printf("P90延迟: %.3f ms\n", p90Latency)
	fmt.Printf("P95延迟: %.3f ms\n", p95Latency)
	fmt.Printf("最小延迟: %.3f ms\n", minLatency)
	fmt.Printf("最大延迟: %.3f ms\n", maxLatency)
	fmt.Printf("吞吐量: %.2f images/sec\n", throughput)
	fmt.Printf("Start PM: %.2f MB\n", startRSS)
	fmt.Printf("Peak PM: %.2f MB\n", peakRSS)
	fmt.Printf("Stable PM: %.2f MB\n", stableRSS)
	fmt.Printf("PM Drift: %.2f MB\n", stableRSS-startRSS)
	fmt.Printf("Go Heap: %.2f MB\n", goHeap)

	// 保存详细日志
	logPath := filepath.Join(basePath, "results", "go_reinforced_detailed_log.txt")
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("创建日志文件失败: %v\n", err)
		return
	}
	defer logFile.Close()

	for i, r := range results {
		fmt.Fprintf(logFile, "===== 第 %d 轮测试 =====\n", i+1)
		fmt.Fprintf(logFile, "平均延迟: %.5f ms\n", r.AvgLatency)
		fmt.Fprintf(logFile, "标准差: %.5f ms\n", r.StdLatency)
		fmt.Fprintf(logFile, "P50延迟: %.5f ms\n", r.P50Latency)
		fmt.Fprintf(logFile, "P90延迟: %.5f ms\n", r.P90Latency)
		fmt.Fprintf(logFile, "P95延迟: %.5f ms\n", r.P95Latency)
		fmt.Fprintf(logFile, "最小延迟: %.5f ms\n", r.MinLatency)
		fmt.Fprintf(logFile, "最大延迟: %.5f ms\n", r.MaxLatency)
		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Fprintf(logFile, "Start PM: %.5f MB\n", r.StartRSS)
		fmt.Fprintf(logFile, "Peak PM: %.5f MB\n", r.PeakRSS)
		fmt.Fprintf(logFile, "Stable PM: %.5f MB\n", r.StableRSS)
		fmt.Fprintf(logFile, "PM Drift: %.5f MB\n", r.StableRSS-r.StartRSS)
		fmt.Fprintf(logFile, "Go Heap: %.5f MB\n", r.GoHeap)
		fmt.Fprintf(logFile, "\n")
	}

	fmt.Fprintf(logFile, "===== 10轮测试平均值 =====\n")
	fmt.Fprintf(logFile, "平均延迟: %.5f ms\n", avgLatency)
	fmt.Fprintf(logFile, "标准差: %.5f ms\n", stdLatency)
	fmt.Fprintf(logFile, "P50延迟: %.5f ms\n", p50Latency)
	fmt.Fprintf(logFile, "P90延迟: %.5f ms\n", p90Latency)
	fmt.Fprintf(logFile, "P95延迟: %.5f ms\n", p95Latency)
	fmt.Fprintf(logFile, "最小延迟: %.5f ms\n", minLatency)
	fmt.Fprintf(logFile, "最大延迟: %.5f ms\n", maxLatency)
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(logFile, "Start PM: %.5f MB\n", startRSS)
	fmt.Fprintf(logFile, "Peak PM: %.5f MB\n", peakRSS)
	fmt.Fprintf(logFile, "Stable PM: %.5f MB\n", stableRSS)
	fmt.Fprintf(logFile, "PM Drift: %.5f MB\n", stableRSS-startRSS)
	fmt.Fprintf(logFile, "Go Heap: %.5f MB\n", goHeap)

	fmt.Printf("\n详细日志已保存到: %s\n", logPath)

	// 保存平均值结果
	resultPath := filepath.Join(basePath, "results", "go_reinforced_result.txt")
	resultFile, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer resultFile.Close()

	fmt.Fprintf(resultFile, "===== Go 强化测试结果（10轮运行） =====\n")
	for i, r := range results {
		fmt.Fprintf(resultFile, "第%d轮平均延迟: %.5f ms\n", i+1, r.AvgLatency)
	}
	fmt.Fprintf(resultFile, "\n===== 10轮测试平均值 =====\n")
	fmt.Fprintf(resultFile, "平均延迟: %.5f ms\n", avgLatency)
	fmt.Fprintf(resultFile, "标准差: %.5f ms\n", stdLatency)
	fmt.Fprintf(resultFile, "P50延迟: %.5f ms\n", p50Latency)
	fmt.Fprintf(resultFile, "P90延迟: %.5f ms\n", p90Latency)
	fmt.Fprintf(resultFile, "P95延迟: %.5f ms\n", p95Latency)
	fmt.Fprintf(resultFile, "\n===== 内存使用情况（10轮运行平均值） =====\n")
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(resultFile, "Start PM: %.5f MB\n", startRSS)
	fmt.Fprintf(resultFile, "Peak PM: %.5f MB\n", peakRSS)
	fmt.Fprintf(resultFile, "Stable PM: %.5f MB\n", stableRSS)
	fmt.Fprintf(resultFile, "PM Drift: %.5f MB\n", stableRSS-startRSS)
	fmt.Fprintf(resultFile, "Go Heap: %.5f MB\n", goHeap)

	fmt.Printf("结果已保存到: %s\n", resultPath)

	// 保存原始延迟数据（用于分析）
	latencyDataPath := filepath.Join(basePath, "results", "go_reinforced_latency_data.txt")
	latencyFile, err := os.Create(latencyDataPath)
	if err != nil {
		fmt.Printf("创建延迟数据文件失败: %v\n", err)
		return
	}
	defer latencyFile.Close()

	for i, r := range results {
		fmt.Fprintf(latencyFile, "===== 第 %d 轮测试 =====\n", i+1)
		for _, t := range r.Times {
			fmt.Fprintf(latencyFile, "%.5f\n", t)
		}
	}

	fmt.Printf("原始延迟数据已保存到: %s\n", latencyDataPath)
	fmt.Println("测试完成!")
}
