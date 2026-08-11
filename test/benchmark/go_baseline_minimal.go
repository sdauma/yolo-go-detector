// go_baseline_minimal.go
// Go 基准测试 - Baseline 执行路径
//
// 重要声明（P0原则）：
// 本测试使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
// 自动启用 I/O Binding。根据 P0 原则，本测试仅用于观察现象，不用于语言级性能结论。
//
// 技术限制说明：
// - Go baseline Session 接口（NewSession）的签名不接受 SessionOptions 参数，
//   线程配置由 ONNX Runtime 的默认行为决定（intra_op_num_threads 默认等于 CPU 核数）
// - 代码中创建了 SessionOptions 并设置了 intraOp=12，但由于 NewSession 不接受 opts，
//   这些设置实际上不生效。保留 opts 创建代码仅用于记录意图
// - 因此，Go 和 Python 的线程配置测试结果不可直接对比
//
// 测试目的：
// - 在 ONNX Runtime 默认线程配置下测量 Go 端的基准性能
// - 为与其他测试的对比提供参考基线
// - 不用于语言级线程扩展性结论

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
	P50Latency float64
	P90Latency float64
	P99Latency float64
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
	for i := 0; i < 20; i++ {
		if err := session.Run(); err != nil {
			return nil, fmt.Errorf("Warmup 运行失败: %v", err)
		}
	}

	// Benchmark
	runs := 200
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
	p99_latency := times[int(float64(runs)*0.99)]

	// 获取 Go heap 内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return &BenchmarkResult{
		AvgLatency: avg_latency,
		P50Latency: p50_latency,
		P90Latency: p90_latency,
		P99Latency: p99_latency,
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
	// 强制设置 12 线程
	runtime.GOMAXPROCS(12)

	// 强制垃圾回收，避免影响测试
	runtime.GC()

	fmt.Println("===== Go 基准测试（10次运行） =====")
	fmt.Printf("Go 线程数设置: %d\n", runtime.GOMAXPROCS(0))

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
		// 每次测试前强制垃圾回收
		runtime.GC()
		fmt.Printf("\n===== 第 %d 次测试 =====\n", i+1)
		result, err := runBenchmark()
		if err != nil {
			fmt.Printf("测试失败: %v\n", err)
			return
		}
		results[i] = result

		fmt.Printf("平均延迟: %.5f ms\n", result.AvgLatency)
		fmt.Printf("P50延迟: %.5f ms\n", result.P50Latency)
		fmt.Printf("P90延迟: %.5f ms\n", result.P90Latency)
		fmt.Printf("P99延迟: %.5f ms\n", result.P99Latency)
		fmt.Printf("最小延迟: %.5f ms\n", result.MinLatency)
		fmt.Printf("最大延迟: %.5f ms\n", result.MaxLatency)
		// 中间数据保留5位小数，符合核心期刊规范
		fmt.Printf("Start PM: %.5f MB\n", result.StartRSS)
		fmt.Printf("Peak PM: %.5f MB\n", result.PeakRSS)
		fmt.Printf("Stable PM: %.5f MB\n", result.StableRSS)
		fmt.Printf("PM Drift: %.5f MB\n", result.StableRSS-result.StartRSS)
		fmt.Printf("Go Heap: %.5f MB\n", result.GoHeap)
	}

	// 计算平均值
	var avgLatency, p50Latency, p90Latency, p99Latency float64
	var minLatency, maxLatency float64
	var startRSS, peakRSS, stableRSS, goHeap float64

	for _, r := range results {
		avgLatency += r.AvgLatency
		p50Latency += r.P50Latency
		p90Latency += r.P90Latency
		p99Latency += r.P99Latency
		minLatency += r.MinLatency
		maxLatency += r.MaxLatency
		startRSS += r.StartRSS
		peakRSS += r.PeakRSS
		stableRSS += r.StableRSS
		goHeap += r.GoHeap
	}

	avgLatency /= float64(numRuns)
	p50Latency /= float64(numRuns)
	p90Latency /= float64(numRuns)
	p99Latency /= float64(numRuns)
	minLatency /= float64(numRuns)
	maxLatency /= float64(numRuns)
	startRSS /= float64(numRuns)
	peakRSS /= float64(numRuns)
	stableRSS /= float64(numRuns)
	goHeap /= float64(numRuns)

	fmt.Printf("\n===== 10 次测试平均值 =====\n")
	fmt.Printf("平均延迟：%.5f ms\n", avgLatency)
	fmt.Printf("P50 延迟：%.5f ms\n", p50Latency)
	fmt.Printf("P90 延迟：%.5f ms\n", p90Latency)
	fmt.Printf("P99 延迟：%.5f ms\n", p99Latency)
	fmt.Printf("最小延迟：%.5f ms\n", minLatency)
	fmt.Printf("最大延迟：%.5f ms\n", maxLatency)
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Printf("Start PM: %.5f MB\n", startRSS)
	fmt.Printf("Peak PM: %.5f MB\n", peakRSS)
	fmt.Printf("Stable PM: %.5f MB\n", stableRSS)
	fmt.Printf("PM Drift: %.5f MB\n", stableRSS-startRSS)
	fmt.Printf("Go Heap: %.5f MB\n", goHeap)

	// 保存详细日志
	logPath := filepath.Join(basePath, "results", "go_baseline_detailed_log.txt")
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("创建日志文件失败: %v\n", err)
		return
	}
	defer logFile.Close()

	for i, r := range results {
		fmt.Fprintf(logFile, "===== 第 %d 次测试 =====\n", i+1)
		fmt.Fprintf(logFile, "平均延迟: %.5f ms\n", r.AvgLatency)
		fmt.Fprintf(logFile, "P50延迟: %.5f ms\n", r.P50Latency)
		fmt.Fprintf(logFile, "P90延迟: %.5f ms\n", r.P90Latency)
		fmt.Fprintf(logFile, "P99延迟: %.5f ms\n", r.P99Latency)
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

	fmt.Fprintf(logFile, "===== 10 次测试平均值 =====\n")
	fmt.Fprintf(logFile, "平均延迟：%.5f ms\n", avgLatency)
	fmt.Fprintf(logFile, "P50 延迟：%.5f ms\n", p50Latency)
	fmt.Fprintf(logFile, "P90 延迟：%.5f ms\n", p90Latency)
	fmt.Fprintf(logFile, "P99 延迟：%.5f ms\n", p99Latency)
	fmt.Fprintf(logFile, "最小延迟：%.5f ms\n", minLatency)
	fmt.Fprintf(logFile, "最大延迟：%.5f ms\n", maxLatency)
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(logFile, "Start PM: %.5f MB\n", startRSS)
	fmt.Fprintf(logFile, "Peak PM: %.5f MB\n", peakRSS)
	fmt.Fprintf(logFile, "Stable PM: %.5f MB\n", stableRSS)
	fmt.Fprintf(logFile, "PM Drift: %.5f MB\n", stableRSS-startRSS)
	fmt.Fprintf(logFile, "Go Heap: %.5f MB\n", goHeap)

	fmt.Printf("\n详细日志已保存到: %s\n", logPath)

	// 保存平均值结果
	resultPath := filepath.Join(basePath, "results", "go_baseline_result.txt")
	resultFile, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer resultFile.Close()

	fmt.Fprintf(resultFile, "===== Go 基准测试结果（10 次运行平均值） =====\n")
	fmt.Fprintf(resultFile, "平均延迟：%.5f ms\n", avgLatency)
	fmt.Fprintf(resultFile, "P50 延迟：%.5f ms\n", p50Latency)
	fmt.Fprintf(resultFile, "P90 延迟：%.5f ms\n", p90Latency)
	fmt.Fprintf(resultFile, "P99 延迟：%.5f ms\n", p99Latency)
	fmt.Fprintf(resultFile, "最小延迟：%.5f ms\n", minLatency)
	fmt.Fprintf(resultFile, "最大延迟：%.5f ms\n", maxLatency)
	fmt.Fprintf(resultFile, "\n===== 内存使用情况（10 次运行平均值） =====\n")
	// 中间数据保留5位小数，符合核心期刊规范
	fmt.Fprintf(resultFile, "Start PM: %.5f MB\n", startRSS)
	fmt.Fprintf(resultFile, "Peak PM: %.5f MB\n", peakRSS)
	fmt.Fprintf(resultFile, "Stable PM: %.5f MB\n", stableRSS)
	fmt.Fprintf(resultFile, "PM Drift: %.5f MB\n", stableRSS-startRSS)
	fmt.Fprintf(resultFile, "Go Heap: %.5f MB\n", goHeap)

	fmt.Printf("结果已保存到: %s\n", resultPath)

	// 保存最后一次测试的原始延迟数据（用于生成箱线图）
	latencyDataPath := filepath.Join(basePath, "results", "go_baseline_latency_data.txt")
	latencyFile, err := os.Create(latencyDataPath)
	if err != nil {
		fmt.Printf("创建延迟数据文件失败: %v\n", err)
		return
	}
	defer latencyFile.Close()

	for _, t := range results[numRuns-1].Times {
		fmt.Fprintf(latencyFile, "%.5f\n", t)
	}

	fmt.Printf("原始延迟数据已保存到: %s\n", latencyDataPath)
	fmt.Println("测试完成!")
}
