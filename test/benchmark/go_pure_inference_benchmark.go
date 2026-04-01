// go_pure_inference_benchmark.go
// Go 纯推理测试 - 输入只加载一次，循环复用
//
// 测试目的：
// - 测量纯推理延迟（不包含IO开销）
// - 输入数据只加载一次，循环复用
// - 确保测试结果反映真实推理性能

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
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// getProcessRSS 获取进程的 RSS（Working Set）内存使用量（MB）
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

func runBenchmark(modelName, modelPath, libPath string) BenchmarkResult {
	fmt.Printf("===== Go 纯推理测试 - %s =====\n", modelName)
	fmt.Println("CPU核心调度：系统默认")

	// 初始化ORT
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 创建会话选项
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("错误: 创建会话选项失败: %v\n", err)
		os.Exit(1)
	}
	defer opts.Destroy()

	// 线程配置 - 12线程
	opts.SetIntraOpNumThreads(12)
	opts.SetInterOpNumThreads(1)

	// 日志配置（关闭所有日志，避免日志IO干扰性能）
	opts.SetLogSeverityLevel(3)

	// 性能分析配置（关闭性能分析，避免额外开销）
	opts.SetExecutionMode(0)

	// 内存池配置（启用内存池复用）
	opts.SetGraphOptimizationLevel(3)

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("错误: 创建输入张量失败: %v\n", err)
		os.Exit(1)
	}
	defer inputTensor.Destroy()

	// 从预生成的二进制文件加载输入数据
	basePath := filepath.Dir(filepath.Dir(modelPath))
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	inputDataFile, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("错误: 读取输入数据文件失败: %v\n", err)
		os.Exit(1)
	}

	// 转换为 float32 并填充到张量
	inputData := inputTensor.GetData()
	expectedSize := 1 * 3 * 640 * 640 * 4 // float32 = 4 bytes
	if len(inputDataFile) != expectedSize {
		fmt.Printf("错误: 输入数据文件大小不匹配: 期望 %d 字节，实际 %d 字节\n", expectedSize, len(inputDataFile))
		os.Exit(1)
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
		fmt.Printf("错误: 创建输出张量失败: %v\n", err)
		os.Exit(1)
	}
	defer outputTensor.Destroy()

	// 创建会话
	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		fmt.Printf("错误: 创建会话失败: %v\n", err)
		os.Exit(1)
	}
	defer session.Destroy()

	fmt.Println("InferenceSession 创建成功!")
	fmt.Printf("输入形状: %v\n", inputShape)
	fmt.Printf("输入数据加载成功: %s\n", inputDataPath)

	// 内存采样点 1：Session 创建后、warmup 前（Start RSS）
	startRSS := getProcessRSS()

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 20; i++ {
		if err := session.Run(); err != nil {
			fmt.Printf("错误: Warmup 运行失败: %v\n", err)
			os.Exit(1)
		}
	}

	// Benchmark - 纯推理，输入复用
	fmt.Println("Running pure inference benchmark...")
	runs := 2000 // 2000次推理
	times := make([]float64, runs)
	peakRSS := startRSS

	for i := 0; i < runs; i++ {
		// 每次推理都创建新的input tensor副本，避免CPU cache效应
		inputTensorCopy, err := ort.NewEmptyTensor[float32](inputShape)
		if err != nil {
			fmt.Printf("错误: 创建输入张量副本失败: %v\n", err)
			os.Exit(1)
		}
		copy(inputTensorCopy.GetData(), inputData)

		start := time.Now()
		if err := session.Run(); err != nil {
			fmt.Printf("错误: 运行失败: %v\n", err)
			inputTensorCopy.Destroy()
			os.Exit(1)
		}
		elapsed := time.Since(start).Milliseconds()
		times[i] = float64(elapsed)

		inputTensorCopy.Destroy()

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
	sum := 0.0
	for _, t := range times {
		sum += t
	}
	avgLatency := sum / float64(len(times))

	// 计算标准差
	variance := 0.0
	for _, t := range times {
		variance += math.Pow(t-avgLatency, 2)
	}
	stdLatency := math.Sqrt(variance / float64(len(times)))

	minLatency := times[0]
	maxLatency := times[len(times)-1]
	p50Latency := times[int(float64(len(times))*0.5)]
	p90Latency := times[int(float64(len(times))*0.9)]
	p95Latency := times[int(float64(len(times))*0.95)]

	// 获取 Go heap 内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return BenchmarkResult{
		AvgLatency: avgLatency,
		StdLatency: stdLatency,
		P50Latency: p50Latency,
		P90Latency: p90Latency,
		P95Latency: p95Latency,
		MinLatency: minLatency,
		MaxLatency: maxLatency,
		StartRSS:   startRSS,
		PeakRSS:    peakRSS,
		StableRSS:  stableRSS,
		GoHeap:     float64(m.Alloc) / 1024 / 1024,
		Times:      times,
	}
}

func main() {
	fmt.Println("===== Go 纯推理测试 ====")
	fmt.Println("测试配置：")
	fmt.Println("- 线程数: 12")
	fmt.Println("- 输入数据: 只加载一次，每次推理创建副本")
	fmt.Println("- 推理次数: 2000次")
	fmt.Println("- Warmup: 20次")
	fmt.Println()

	// 获取项目根路径
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(currentDir))

	// 设置模型和库路径
	yolo11xPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	yolo11nPath := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")

	// 检查文件是否存在
	if !fileExists(yolo11xPath) {
		fmt.Printf("错误: YOLO11x模型文件不存在: %s\n", yolo11xPath)
		os.Exit(1)
	}
	if !fileExists(yolo11nPath) {
		fmt.Printf("错误: YOLO11n模型文件不存在: %s\n", yolo11nPath)
		os.Exit(1)
	}
	if !fileExists(libPath) {
		fmt.Printf("错误: 库文件不存在: %s\n", libPath)
		os.Exit(1)
	}

	// 禁用GC，避免GC干扰性能测试
	// debug.SetGCPercent(-1)

	// 测试YOLO11x
	resultX := runBenchmark("YOLO11x", yolo11xPath, libPath)
	fmt.Printf("\n===== YOLO11x 结果 =====\n")
	fmt.Printf("平均延迟: %.5f ms\n", resultX.AvgLatency)
	fmt.Printf("标准差: %.5f ms\n", resultX.StdLatency)
	fmt.Printf("P50延迟: %.5f ms\n", resultX.P50Latency)
	fmt.Printf("P90延迟: %.5f ms\n", resultX.P90Latency)
	fmt.Printf("P95延迟: %.5f ms\n", resultX.P95Latency)
	fmt.Printf("最小延迟: %.5f ms\n", resultX.MinLatency)
	fmt.Printf("最大延迟: %.5f ms\n", resultX.MaxLatency)
	fmt.Printf("Start RSS: %.5f MB\n", resultX.StartRSS)
	fmt.Printf("Peak RSS: %.5f MB\n", resultX.PeakRSS)
	fmt.Printf("Stable RSS: %.5f MB\n", resultX.StableRSS)
	fmt.Printf("RSS Drift: %.5f MB\n", resultX.StableRSS-resultX.StartRSS)
	fmt.Printf("Go Heap: %.5f MB\n", resultX.GoHeap)

	// 测试YOLO11n
	resultN := runBenchmark("YOLO11n", yolo11nPath, libPath)
	fmt.Printf("\n===== YOLO11n 结果 =====\n")
	fmt.Printf("平均延迟: %.5f ms\n", resultN.AvgLatency)
	fmt.Printf("标准差: %.5f ms\n", resultN.StdLatency)
	fmt.Printf("P50延迟: %.5f ms\n", resultN.P50Latency)
	fmt.Printf("P90延迟: %.5f ms\n", resultN.P90Latency)
	fmt.Printf("P95延迟: %.5f ms\n", resultN.P95Latency)
	fmt.Printf("最小延迟: %.5f ms\n", resultN.MinLatency)
	fmt.Printf("最大延迟: %.5f ms\n", resultN.MaxLatency)
	fmt.Printf("Start RSS: %.5f MB\n", resultN.StartRSS)
	fmt.Printf("Peak RSS: %.5f MB\n", resultN.PeakRSS)
	fmt.Printf("Stable RSS: %.5f MB\n", resultN.StableRSS)
	fmt.Printf("RSS Drift: %.5f MB\n", resultN.StableRSS-resultN.StartRSS)
	fmt.Printf("Go Heap: %.5f MB\n", resultN.GoHeap)

	// 保存结果
	resultsDir := filepath.Join(basePath, "results")
	os.MkdirAll(resultsDir, 0755)

	resultPath := filepath.Join(resultsDir, "go_pure_inference_result.txt")
	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	fmt.Fprintf(file, "===== Go 纯推理测试结果 =====\n")
	fmt.Fprintf(file, "\n===== YOLO11x =====\n")
	fmt.Fprintf(file, "平均延迟: %.5f ms\n", resultX.AvgLatency)
	fmt.Fprintf(file, "标准差: %.5f ms\n", resultX.StdLatency)
	fmt.Fprintf(file, "P50延迟: %.5f ms\n", resultX.P50Latency)
	fmt.Fprintf(file, "P90延迟: %.5f ms\n", resultX.P90Latency)
	fmt.Fprintf(file, "P95延迟: %.5f ms\n", resultX.P95Latency)
	fmt.Fprintf(file, "最小延迟: %.5f ms\n", resultX.MinLatency)
	fmt.Fprintf(file, "最大延迟: %.5f ms\n", resultX.MaxLatency)
	fmt.Fprintf(file, "Start RSS: %.5f MB\n", resultX.StartRSS)
	fmt.Fprintf(file, "Peak RSS: %.5f MB\n", resultX.PeakRSS)
	fmt.Fprintf(file, "Stable RSS: %.5f MB\n", resultX.StableRSS)
	fmt.Fprintf(file, "RSS Drift: %.5f MB\n", resultX.StableRSS-resultX.StartRSS)
	fmt.Fprintf(file, "Go Heap: %.5f MB\n", resultX.GoHeap)

	fmt.Fprintf(file, "\n===== YOLO11n =====\n")
	fmt.Fprintf(file, "平均延迟: %.5f ms\n", resultN.AvgLatency)
	fmt.Fprintf(file, "标准差: %.5f ms\n", resultN.StdLatency)
	fmt.Fprintf(file, "P50延迟: %.5f ms\n", resultN.P50Latency)
	fmt.Fprintf(file, "P90延迟: %.5f ms\n", resultN.P90Latency)
	fmt.Fprintf(file, "P95延迟: %.5f ms\n", resultN.P95Latency)
	fmt.Fprintf(file, "最小延迟: %.5f ms\n", resultN.MinLatency)
	fmt.Fprintf(file, "最大延迟: %.5f ms\n", resultN.MaxLatency)
	fmt.Fprintf(file, "Start RSS: %.5f MB\n", resultN.StartRSS)
	fmt.Fprintf(file, "Peak RSS: %.5f MB\n", resultN.PeakRSS)
	fmt.Fprintf(file, "Stable RSS: %.5f MB\n", resultN.StableRSS)
	fmt.Fprintf(file, "RSS Drift: %.5f MB\n", resultN.StableRSS-resultN.StartRSS)
	fmt.Fprintf(file, "Go Heap: %.5f MB\n", resultN.GoHeap)

	fmt.Printf("\n结果已保存到: %s\n", resultPath)
	fmt.Println("测试完成!")
}
