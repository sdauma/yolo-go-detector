// thread_config_benchmark.go
// Go 线程配置性能测试 - Baseline 执行路径
//
// 重要声明（P0原则）：
// 本测试使用 Go baseline Session 接口（NewSession），由于技术限制，实际上启用了 I/O Binding。
// 根据 P0 原则，本测试仅用于观察现象，不用于语言级性能结论。
//
// 技术限制说明：
// - Go baseline Session 接口（NewSession）不支持显式设置线程参数
// - 线程配置可能依赖 ONNX Runtime 的默认行为
// - 因此，Go 和 Python 的线程配置测试结果不可直接对比
//
// 测试目的：
// - 观察不同线程配置下的性能趋势
// - 验证 ONNX Runtime 的线程扩展性
// - 不用于语言级线程扩展性结论
//
// 测试线程配置: 1, 2, 4, 8

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
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
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

// loadInputDataFromFile 从二进制文件加载输入数据
func loadInputDataFromFile(data []float32, filePath string) error {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("打开文件失败: %w", err)
	}
	defer file.Close()

	// 读取文件内容到缓冲区
	buffer := make([]byte, len(data)*4) // float32 占 4 字节
	_, err = file.Read(buffer)
	if err != nil {
		return fmt.Errorf("读取文件失败: %w", err)
	}

	// 将字节数据转换为 float32（使用 LittleEndian 字节序）
	for i := 0; i < len(data); i++ {
		offset := i * 4
		// 读取 4 字节并转换为 uint32
		u32 := binary.LittleEndian.Uint32(buffer[offset : offset+4])
		// 将 uint32 转换为 float32
		data[i] = *(*float32)(unsafe.Pointer(&u32))
	}

	return nil
}

// ThreadConfigResult 线程配置测试结果
type ThreadConfigResult struct {
	IntraOpNumThreads int     `json:"intra_op_num_threads"`
	AvgLatency        float64 `json:"avg_latency"`
	MinLatency        float64 `json:"min_latency"`
	MaxLatency        float64 `json:"max_latency"`
	P50Latency        float64 `json:"p50_latency"`
	P90Latency        float64 `json:"p90_latency"`
	P99Latency        float64 `json:"p99_latency"`
	StdDevLatency     float64 `json:"std_dev_latency"`
	CoeffVarLatency   float64 `json:"coeff_var_latency"`
	FPS               float64 `json:"fps"`
	StartRSS          float64 `json:"start_rss"`
	PeakRSS           float64 `json:"peak_rss"`
	StableRSS         float64 `json:"stable_rss"`
}

// calculateStdDev 计算标准差
func calculateStdDev(values []float64, mean float64) float64 {
	var sumSquaredDiff float64
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	return math.Sqrt(sumSquaredDiff / float64(len(values)))
}

func main() {
	fmt.Println("===== 不同 intra_op_num_threads 配置性能测试 =====")

	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取工作目录失败: %v\n", err)
		return
	}
	fmt.Printf("当前目录: %s\n", wd)

	// 构建项目根路径
	basePath := filepath.Dir(filepath.Dir(wd))
	fmt.Printf("项目根路径: %s\n", basePath)

	// 设置模型和库路径
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")

	fmt.Printf("模型路径: %s\n", modelPath)
	fmt.Printf("库路径: %s\n", libPath)

	// 检查文件是否存在
	if !fileExists(modelPath) {
		fmt.Printf("错误: 模型文件不存在: %s\n", modelPath)
		return
	}
	if !fileExists(libPath) {
		fmt.Printf("错误: 库文件不存在: %s\n", libPath)
		return
	}

	// 初始化ORT
	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化 ONNX Runtime 环境失败: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()
	fmt.Println("ONNX Runtime 环境初始化成功!")

	// 测试不同的线程配置
	threadConfigs := []int{1, 2, 4, 8}
	results := make([]ThreadConfigResult, 0, len(threadConfigs))

	for _, numThreads := range threadConfigs {
		fmt.Printf("\n===== 测试线程配置: intra_op_num_threads=%d =====\n", numThreads)

		// 执行5次独立测试
		testCount := 5
		var allAvgLatencies []float64
		var allMinLatencies []float64
		var allMaxLatencies []float64
		var allP50Latencies []float64
		var allP90Latencies []float64
		var allP99Latencies []float64
		var allStartRSS []float64
		var allPeakRSS []float64
		var allStableRSS []float64

		for testIdx := 1; testIdx <= testCount; testIdx++ {
			fmt.Printf("\n--- 独立测试 %d/%d --->\n", testIdx, testCount)

			// 创建会话选项
			opts, err := ort.NewSessionOptions()
			if err != nil {
				fmt.Printf("创建会话选项失败: %v\n", err)
				continue
			}

			// 显式设置所有 SessionOptions 参数（P2原则：禁止依赖默认值）
			// 线程配置
			opts.SetIntraOpNumThreads(numThreads)
			opts.SetInterOpNumThreads(1)

			// 日志配置（关闭所有日志，避免日志IO干扰性能）
			opts.SetLogSeverityLevel(3) // 3 = ORT_LOGGING_LEVEL_ERROR

			// 性能分析配置（关闭性能分析，避免额外开销）
			opts.SetExecutionMode(0) // 0 = ORT_SEQUENTIAL

			// 内存池配置（启用内存池复用）
			opts.SetGraphOptimizationLevel(3) // 3 = ORT_ENABLE_ALL

			// 所有未提及的Session参数均使用ONNX Runtime 1.23.2官方默认值

			// 创建输入张量
			inputShape := ort.NewShape(1, 3, 640, 640)
			inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
			if err != nil {
				fmt.Printf("创建输入张量失败: %v\n", err)
				opts.Destroy()
				continue
			}

			// 准备输入数据（从文件加载，确保与 Python 版本一致）
			fmt.Println("加载输入数据...")
			inputData := inputTensor.GetData()
			// 计算数据文件路径
			inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
			err = loadInputDataFromFile(inputData, inputDataPath)
			if err != nil {
				fmt.Printf("加载输入数据失败: %v\n", err)
				inputTensor.Destroy()
				opts.Destroy()
				continue
			}
			fmt.Printf("输入数据加载成功: %s\n", inputDataPath)

			// 创建会话（使用默认执行路径，绑定输入和输出张量）
			fmt.Println("创建 Session...")
			// 创建输出张量（YOLO11x 的输出形状通常为 [1, 84, 8400]）
			outputShape := ort.NewShape(1, 84, 8400)
			outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
			if err != nil {
				fmt.Printf("创建输出张量失败: %v\n", err)
				inputTensor.Destroy()
				opts.Destroy()
				continue
			}

			// 使用与 Python 相同的默认执行路径
			session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
			if err != nil {
				fmt.Printf("创建会话失败: %v\n", err)
				inputTensor.Destroy()
				outputTensor.Destroy()
				opts.Destroy()
				continue
			}

			// 验证线程配置
			fmt.Printf("测试线程配置: intra=%d, inter=%d\n", numThreads, 1)

			// 内存采样点 1：Session 创建后、warmup 前（Start RSS）
			startRSS := getProcessRSS()
			fmt.Printf("Start RSS: %.2f MB\n", startRSS)

			// Warmup
			fmt.Println("Warming up...")
			for i := 0; i < 10; i++ {
				err := session.Run()
				if err != nil {
					fmt.Printf("Warmup 运行失败: %v\n", err)
					session.Destroy()
					inputTensor.Destroy()
					outputTensor.Destroy()
					opts.Destroy()
					continue
				}
			}

			// 内存采样点 2：Warmup 后
			warmupRSS := getProcessRSS()
			fmt.Printf("Warmup RSS: %.2f MB\n", warmupRSS)

			// Benchmark
			fmt.Println("Running benchmark...")
			runs := 100
			var sum float64
			times := make([]float64, runs)
			peakRSS := startRSS

			// 每10次推理采样一次内存，减少开销
			for i := 0; i < runs; i++ {
				t0 := time.Now()
				err := session.Run()
				if err != nil {
					fmt.Printf("运行失败: %v\n", err)
					session.Destroy()
					inputTensor.Destroy()
					outputTensor.Destroy()
					opts.Destroy()
					continue
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
			fmt.Printf("Stable RSS: %.2f MB\n", stableRSS)

			// 计算结果
			sort.Float64s(times)
			avg_latency := sum / float64(runs)
			min_latency := times[0]
			max_latency := times[runs-1]
			p50_latency := times[runs/2]
			p90_latency := times[int(float64(runs)*0.9)]
			p99_latency := times[int(float64(runs)*0.99)]

			// 保存本次测试结果
			allAvgLatencies = append(allAvgLatencies, avg_latency)
			allMinLatencies = append(allMinLatencies, min_latency)
			allMaxLatencies = append(allMaxLatencies, max_latency)
			allP50Latencies = append(allP50Latencies, p50_latency)
			allP90Latencies = append(allP90Latencies, p90_latency)
			allP99Latencies = append(allP99Latencies, p99_latency)
			allStartRSS = append(allStartRSS, startRSS)
			allPeakRSS = append(allPeakRSS, peakRSS)
			allStableRSS = append(allStableRSS, stableRSS)

			fmt.Printf("测试 %d 完成: 平均延迟=%.3f ms\n", testIdx, avg_latency)

			// 释放资源
			session.Destroy()
			inputTensor.Destroy()
			outputTensor.Destroy()
			opts.Destroy()
		}

		// 计算3次测试的平均值
		var totalAvgLatency, totalMinLatency, totalMaxLatency, totalP50Latency, totalP90Latency, totalP99Latency float64
		var totalStartRSS, totalPeakRSS, totalStableRSS float64
		for i := 0; i < len(allAvgLatencies); i++ {
			totalAvgLatency += allAvgLatencies[i]
			totalMinLatency += allMinLatencies[i]
			totalMaxLatency += allMaxLatencies[i]
			totalP50Latency += allP50Latencies[i]
			totalP90Latency += allP90Latencies[i]
			totalP99Latency += allP99Latencies[i]
			totalStartRSS += allStartRSS[i]
			totalPeakRSS += allPeakRSS[i]
			totalStableRSS += allStableRSS[i]
		}
		testCountFloat := float64(len(allAvgLatencies))
		avgLatency := totalAvgLatency / testCountFloat
		minLatency := totalMinLatency / testCountFloat
		maxLatency := totalMaxLatency / testCountFloat
		p50Latency := totalP50Latency / testCountFloat
		p90Latency := totalP90Latency / testCountFloat
		p99Latency := totalP99Latency / testCountFloat
		startRSS := totalStartRSS / testCountFloat
		peakRSS := totalPeakRSS / testCountFloat
		stableRSS := totalStableRSS / testCountFloat

		// 计算标准差
		stdDevLatency := calculateStdDev(allAvgLatencies, avgLatency)
		// 计算变异系数
		coeffVarLatency := stdDevLatency / avgLatency * 100
		// 计算FPS
		fps := 1000.0 / avgLatency

		// 获取 Go heap 内存使用情况（辅助指标）
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		fmt.Printf("\n===== 测试结果 =====\n")
		fmt.Printf("平均延迟: %.3f ms\n", avgLatency)
		fmt.Printf("标准差: %.3f ms\n", stdDevLatency)
		fmt.Printf("变异系数: %.2f%%\n", coeffVarLatency)
		fmt.Printf("FPS: %.2f\n", fps)
		fmt.Printf("P50延迟: %.3f ms\n", p50Latency)
		fmt.Printf("P90延迟: %.3f ms\n", p90Latency)
		fmt.Printf("P99延迟: %.3f ms\n", p99Latency)
		fmt.Printf("最小延迟: %.3f ms\n", minLatency)
		fmt.Printf("最大延迟: %.3f ms\n", maxLatency)
		fmt.Printf("\n===== 内存使用情况 =====\n")
		fmt.Printf("Start RSS: %.2f MB\n", startRSS)
		fmt.Printf("Peak RSS: %.2f MB\n", peakRSS)
		fmt.Printf("Stable RSS: %.2f MB\n", stableRSS)
		fmt.Printf("RSS Drift: %.2f MB\n", stableRSS-startRSS)
		fmt.Printf("Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)

		// 保存结果
		result := ThreadConfigResult{
			IntraOpNumThreads: numThreads,
			AvgLatency:        avgLatency,
			MinLatency:        minLatency,
			MaxLatency:        maxLatency,
			P50Latency:        p50Latency,
			P90Latency:        p90Latency,
			P99Latency:        p99Latency,
			StdDevLatency:     stdDevLatency,
			CoeffVarLatency:   coeffVarLatency,
			FPS:               fps,
			StartRSS:          startRSS,
			PeakRSS:           peakRSS,
			StableRSS:         stableRSS,
		}
		results = append(results, result)

		// 保存详细日志
		logPath := filepath.Join(basePath, "results", fmt.Sprintf("go_thread_%d_detailed_log.txt", numThreads))
		logFile, err := os.Create(logPath)
		if err != nil {
			fmt.Printf("创建日志文件失败: %v\n", err)
		} else {
			for i := 0; i < len(allAvgLatencies); i++ {
				fmt.Fprintf(logFile, "===== 第 %d 次测试 =====\n", i+1)
				fmt.Fprintf(logFile, "平均延迟: %.3f ms\n", allAvgLatencies[i])
				fmt.Fprintf(logFile, "最小延迟: %.3f ms\n", allMinLatencies[i])
				fmt.Fprintf(logFile, "最大延迟: %.3f ms\n", allMaxLatencies[i])
				fmt.Fprintf(logFile, "P50延迟: %.3f ms\n", allP50Latencies[i])
				fmt.Fprintf(logFile, "P90延迟: %.3f ms\n", allP90Latencies[i])
				fmt.Fprintf(logFile, "P99延迟: %.3f ms\n", allP99Latencies[i])
				fmt.Fprintf(logFile, "Start RSS: %.2f MB\n", allStartRSS[i])
				fmt.Fprintf(logFile, "Peak RSS: %.2f MB\n", allPeakRSS[i])
				fmt.Fprintf(logFile, "Stable RSS: %.2f MB\n", allStableRSS[i])
				fmt.Fprintf(logFile, "\n")
			}

			fmt.Fprintf(logFile, "===== 5次测试平均值 =====\n")
			fmt.Fprintf(logFile, "平均延迟: %.3f ms\n", avgLatency)
			fmt.Fprintf(logFile, "标准差: %.3f ms\n", stdDevLatency)
			fmt.Fprintf(logFile, "变异系数: %.2f%%\n", coeffVarLatency)
			fmt.Fprintf(logFile, "FPS: %.2f\n", fps)
			fmt.Fprintf(logFile, "P50延迟: %.3f ms\n", p50Latency)
			fmt.Fprintf(logFile, "P90延迟: %.3f ms\n", p90Latency)
			fmt.Fprintf(logFile, "P99延迟: %.3f ms\n", p99Latency)
			fmt.Fprintf(logFile, "最小延迟: %.3f ms\n", minLatency)
			fmt.Fprintf(logFile, "最大延迟: %.3f ms\n", maxLatency)
			fmt.Fprintf(logFile, "\n===== 内存使用情况 =====\n")
			fmt.Fprintf(logFile, "Start RSS: %.2f MB\n", startRSS)
			fmt.Fprintf(logFile, "Peak RSS: %.2f MB\n", peakRSS)
			fmt.Fprintf(logFile, "Stable RSS: %.2f MB\n", stableRSS)
			fmt.Fprintf(logFile, "RSS Drift: %.2f MB\n", stableRSS-startRSS)
			fmt.Fprintf(logFile, "Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)

			logFile.Close()
			fmt.Printf("详细日志已保存到: %s\n", logPath)
		}

		// 保存单个配置的结果
		resultPath := filepath.Join(basePath, "results", fmt.Sprintf("go_thread_%d_result.txt", numThreads))
		fmt.Printf("\n保存结果到: %s\n", resultPath)
		file, err := os.Create(resultPath)
		if err != nil {
			fmt.Printf("创建文件失败: %v\n", err)
		} else {
			// 写入结果
			fmt.Fprintf(file, "===== 线程配置测试结果（5次运行平均值）: intra_op_num_threads=%d =====\n", numThreads)
			fmt.Fprintf(file, "平均延迟: %.3f ms\n", avgLatency)
			fmt.Fprintf(file, "标准差: %.3f ms\n", stdDevLatency)
			fmt.Fprintf(file, "变异系数: %.2f%%\n", coeffVarLatency)
			fmt.Fprintf(file, "FPS: %.2f\n", fps)
			fmt.Fprintf(file, "P50延迟: %.3f ms\n", p50Latency)
			fmt.Fprintf(file, "P90延迟: %.3f ms\n", p90Latency)
			fmt.Fprintf(file, "P99延迟: %.3f ms\n", p99Latency)
			fmt.Fprintf(file, "最小延迟: %.3f ms\n", minLatency)
			fmt.Fprintf(file, "最大延迟: %.3f ms\n", maxLatency)
			fmt.Fprintf(file, "\n===== 内存使用情况 =====\n")
			fmt.Fprintf(file, "Start RSS: %.2f MB\n", startRSS)
			fmt.Fprintf(file, "Peak RSS: %.2f MB\n", peakRSS)
			fmt.Fprintf(file, "Stable RSS: %.2f MB\n", stableRSS)
			fmt.Fprintf(file, "RSS Drift: %.2f MB\n", stableRSS-startRSS)
			fmt.Fprintf(file, "Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)
			file.Close()
			fmt.Printf("文件写入成功!\n")
		}
	}

	// 保存所有线程配置的综合结果
	comprehensiveResultPath := filepath.Join(basePath, "results", "go_thread_config_comprehensive.txt")
	fmt.Printf("\n保存综合结果到: %s\n", comprehensiveResultPath)
	file, err := os.Create(comprehensiveResultPath)
	if err != nil {
		fmt.Printf("创建综合结果文件失败: %v\n", err)
	} else {
		// 写入综合结果
		fmt.Fprintf(file, "===== 不同 intra_op_num_threads 配置性能测试综合结果 =====\n\n")
		fmt.Fprintf(file, "%-20s %-15s %-12s %-12s %-10s %-15s %-15s %-15s %-15s %-15s\n",
			"线程配置", "平均延迟(ms)", "标准差(ms)", "变异系数(%)", "FPS", "P50延迟(ms)", "P90延迟(ms)", "P99延迟(ms)", "Start RSS(MB)", "Stable RSS(MB)")

		for _, result := range results {
			fmt.Fprintf(file, "%-20d %-15.3f %-12.3f %-12.2f %-10.2f %-15.3f %-15.3f %-15.3f %-15.2f %-15.2f\n",
				result.IntraOpNumThreads, result.AvgLatency, result.StdDevLatency, result.CoeffVarLatency,
				result.FPS, result.P50Latency, result.P90Latency, result.P99Latency, result.StartRSS, result.StableRSS)
		}
		file.Close()
		fmt.Printf("综合结果文件写入成功!\n")
	}

	fmt.Println("\n===== 所有线程配置测试完成 =====")
}
