// cold_start_benchmark.go
// 测试 YOLO11x 模型的冷启动时间和稳定状态推理时间的对比
// 冷启动时间: 会话创建后第一次推理的时间
// 稳定状态时间: 多次推理后的平均时间

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

// ColdStartResult 冷启动测试结果
type ColdStartResult struct {
	ColdStartLatency float64 `json:"cold_start_latency"`
	AvgStableLatency float64 `json:"avg_stable_latency"`
	MinStableLatency float64 `json:"min_stable_latency"`
	MaxStableLatency float64 `json:"max_stable_latency"`
	P50StableLatency float64 `json:"p50_stable_latency"`
	P90StableLatency float64 `json:"p90_stable_latency"`
	P99StableLatency float64 `json:"p99_stable_latency"`
	StdDevStable     float64 `json:"std_dev_stable"`
	CoeffVarStable   float64 `json:"coeff_var_stable"`
	FPS              float64 `json:"fps"`
	StartRSS         float64 `json:"start_rss"`
	ColdStartRSS     float64 `json:"cold_start_rss"`
	StableRSS        float64 `json:"stable_rss"`
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
	fmt.Println("===== 冷启动时间对比分析测试 =====")

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

	// 执行3次独立测试
	testCount := 3
	var allColdStartTimes []float64
	var allAvgStableLatencies []float64
	var allMinStableLatencies []float64
	var allMaxStableLatencies []float64
	var allP50StableLatencies []float64
	var allP90StableLatencies []float64
	var allP99StableLatencies []float64
	var allStartRSS []float64
	var allColdStartRSS []float64
	var allStableRSS []float64

	for testIdx := 1; testIdx <= testCount; testIdx++ {
		fmt.Printf("\n=== 独立测试 %d/%d ===\n", testIdx, testCount)

		// 创建会话选项
		opts, err := ort.NewSessionOptions()
		if err != nil {
			fmt.Printf("创建会话选项失败: %v\n", err)
			continue
		}

		// 设置线程配置
		opts.SetIntraOpNumThreads(4)
		opts.SetInterOpNumThreads(1)

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

		// 内存采样点 1：Session 创建后（Start RSS）
		startRSS := getProcessRSS()
		fmt.Printf("Start RSS: %.2f MB\n", startRSS)

		// 测试冷启动时间
		fmt.Println("\n===== 测试冷启动时间 =====")
		t0 := time.Now()
		err = session.Run()
		if err != nil {
			fmt.Printf("冷启动运行失败: %v\n", err)
			session.Destroy()
			inputTensor.Destroy()
			outputTensor.Destroy()
			opts.Destroy()
			continue
		}
		coldStartTime := time.Since(t0).Seconds() * 1000.0
		fmt.Printf("冷启动时间: %.3f ms\n", coldStartTime)

		// 内存采样点 2：冷启动后（Cold Start RSS）
		coldStartRSS := getProcessRSS()
		fmt.Printf("Cold Start RSS: %.2f MB\n", coldStartRSS)

		// 预热阶段
		fmt.Println("\n===== 预热阶段 =====")
		warmupCount := 10
		warmupLatencies := make([]float64, warmupCount)
		for i := 0; i < warmupCount; i++ {
			t0 := time.Now()
			err := session.Run()
			if err != nil {
				fmt.Printf("预热运行失败: %v\n", err)
				session.Destroy()
				inputTensor.Destroy()
				outputTensor.Destroy()
				opts.Destroy()
				continue
			}
			dt := time.Since(t0).Seconds() * 1000.0
			warmupLatencies[i] = dt
		}

		// 稳定状态测试
		fmt.Println("\n===== 稳定状态测试 =====")
		stableCount := 100
		stableLatencies := make([]float64, stableCount)
		peakRSS := coldStartRSS

		for i := 0; i < stableCount; i++ {
			t0 := time.Now()
			err := session.Run()
			if err != nil {
				fmt.Printf("稳定状态运行失败: %v\n", err)
				session.Destroy()
				inputTensor.Destroy()
				outputTensor.Destroy()
				opts.Destroy()
				continue
			}
			dt := time.Since(t0).Seconds() * 1000.0
			stableLatencies[i] = dt

			// 每10次推理采样一次内存，记录峰值
			if i%10 == 0 {
				currentRSS := getProcessRSS()
				if currentRSS > peakRSS {
					peakRSS = currentRSS
				}
			}
		}

		// 内存采样点 3：稳定状态后（Stable RSS）
		stableRSS := getProcessRSS()
		fmt.Printf("\nStable RSS: %.2f MB\n", stableRSS)
		fmt.Printf("Peak RSS: %.2f MB\n", peakRSS)

		// 计算稳定状态的统计数据
		sort.Float64s(stableLatencies)
		var sumStable float64
		for _, latency := range stableLatencies {
			sumStable += latency
		}
		avgStableLatency := sumStable / float64(stableCount)
		minStableLatency := stableLatencies[0]
		maxStableLatency := stableLatencies[stableCount-1]
		p50StableLatency := stableLatencies[stableCount/2]
		p90StableLatency := stableLatencies[int(float64(stableCount)*0.9)]
		p99StableLatency := stableLatencies[int(float64(stableCount)*0.99)]

		// 保存本次测试结果
		allColdStartTimes = append(allColdStartTimes, coldStartTime)
		allAvgStableLatencies = append(allAvgStableLatencies, avgStableLatency)
		allMinStableLatencies = append(allMinStableLatencies, minStableLatency)
		allMaxStableLatencies = append(allMaxStableLatencies, maxStableLatency)
		allP50StableLatencies = append(allP50StableLatencies, p50StableLatency)
		allP90StableLatencies = append(allP90StableLatencies, p90StableLatency)
		allP99StableLatencies = append(allP99StableLatencies, p99StableLatency)
		allStartRSS = append(allStartRSS, startRSS)
		allColdStartRSS = append(allColdStartRSS, coldStartRSS)
		allStableRSS = append(allStableRSS, stableRSS)

		fmt.Printf("测试 %d 完成: 冷启动时间=%.3f ms, 稳定状态平均时间=%.3f ms\n", testIdx, coldStartTime, avgStableLatency)

		// 释放资源
		session.Destroy()
		inputTensor.Destroy()
		outputTensor.Destroy()
		opts.Destroy()
	}

	// 计算3次测试的平均值
	var totalColdStartTime, totalAvgStableLatency, totalMinStableLatency, totalMaxStableLatency float64
	var totalP50StableLatency, totalP90StableLatency, totalP99StableLatency float64
	var totalStartRSS, totalColdStartRSS, totalStableRSS float64
	for i := 0; i < len(allColdStartTimes); i++ {
		totalColdStartTime += allColdStartTimes[i]
		totalAvgStableLatency += allAvgStableLatencies[i]
		totalMinStableLatency += allMinStableLatencies[i]
		totalMaxStableLatency += allMaxStableLatencies[i]
		totalP50StableLatency += allP50StableLatencies[i]
		totalP90StableLatency += allP90StableLatencies[i]
		totalP99StableLatency += allP99StableLatencies[i]
		totalStartRSS += allStartRSS[i]
		totalColdStartRSS += allColdStartRSS[i]
		totalStableRSS += allStableRSS[i]
	}
	testCountFloat := float64(len(allColdStartTimes))
	coldStartTime := totalColdStartTime / testCountFloat
	avgStableLatency := totalAvgStableLatency / testCountFloat
	minStableLatency := totalMinStableLatency / testCountFloat
	maxStableLatency := totalMaxStableLatency / testCountFloat
	p50StableLatency := totalP50StableLatency / testCountFloat
	p90StableLatency := totalP90StableLatency / testCountFloat
	p99StableLatency := totalP99StableLatency / testCountFloat
	startRSS := totalStartRSS / testCountFloat
	coldStartRSS := totalColdStartRSS / testCountFloat
	stableRSS := totalStableRSS / testCountFloat

	// 计算标准差
	stdDevStable := calculateStdDev(allAvgStableLatencies, avgStableLatency)
	// 计算变异系数
	coeffVarStable := stdDevStable / avgStableLatency * 100
	// 计算FPS
	fps := 1000.0 / avgStableLatency

	// 获取 Go heap 内存使用情况（辅助指标）
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 输出结果
	fmt.Printf("\n===== 冷启动与稳定状态对比结果 =====\n")
	fmt.Printf("冷启动时间: %.3f ms\n", coldStartTime)
	fmt.Printf("稳定状态平均时间: %.3f ms\n", avgStableLatency)
	fmt.Printf("冷启动时间 / 稳定状态平均时间: %.2f 倍\n", coldStartTime/avgStableLatency)
	fmt.Printf("\n===== 稳定状态详细统计 =====\n")
	fmt.Printf("平均延迟: %.3f ms\n", avgStableLatency)
	fmt.Printf("标准差: %.3f ms\n", stdDevStable)
	fmt.Printf("变异系数: %.2f%%\n", coeffVarStable)
	fmt.Printf("FPS: %.2f\n", fps)
	fmt.Printf("最小延迟: %.3f ms\n", minStableLatency)
	fmt.Printf("最大延迟: %.3f ms\n", maxStableLatency)
	fmt.Printf("P50延迟: %.3f ms\n", p50StableLatency)
	fmt.Printf("P90延迟: %.3f ms\n", p90StableLatency)
	fmt.Printf("P99延迟: %.3f ms\n", p99StableLatency)
	fmt.Printf("\n===== 内存使用情况 =====\n")
	fmt.Printf("Start RSS: %.2f MB\n", startRSS)
	fmt.Printf("Cold Start RSS: %.2f MB\n", coldStartRSS)
	fmt.Printf("Stable RSS: %.2f MB\n", stableRSS)
	fmt.Printf("内存增长 (Start -> Cold Start): %.2f MB\n", coldStartRSS-startRSS)
	fmt.Printf("内存增长 (Cold Start -> Stable): %.2f MB\n", stableRSS-coldStartRSS)
	fmt.Printf("Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)

	// 保存结果
	result := ColdStartResult{
		ColdStartLatency: coldStartTime,
		AvgStableLatency: avgStableLatency,
		MinStableLatency: minStableLatency,
		MaxStableLatency: maxStableLatency,
		P50StableLatency: p50StableLatency,
		P90StableLatency: p90StableLatency,
		P99StableLatency: p99StableLatency,
		StdDevStable:     stdDevStable,
		CoeffVarStable:   coeffVarStable,
		FPS:              fps,
		StartRSS:         startRSS,
		ColdStartRSS:     coldStartRSS,
		StableRSS:        stableRSS,
	}

	// 保存结果到文件
	resultPath := filepath.Join(basePath, "results", "go_cold_start_result.txt")
	fmt.Printf("\n保存结果到: %s\n", resultPath)
	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入结果
	fmt.Fprintf(file, "===== 冷启动时间对比分析测试结果 =====\n\n")
	fmt.Fprintf(file, "冷启动时间: %.3f ms\n", result.ColdStartLatency)
	fmt.Fprintf(file, "稳定状态平均时间: %.3f ms\n", result.AvgStableLatency)
	fmt.Fprintf(file, "冷启动时间 / 稳定状态平均时间: %.2f 倍\n\n", result.ColdStartLatency/result.AvgStableLatency)

	fmt.Fprintf(file, "===== 稳定状态详细统计 =====\n")
	fmt.Fprintf(file, "平均延迟: %.3f ms\n", result.AvgStableLatency)
	fmt.Fprintf(file, "标准差: %.3f ms\n", result.StdDevStable)
	fmt.Fprintf(file, "变异系数: %.2f%%\n", result.CoeffVarStable)
	fmt.Fprintf(file, "FPS: %.2f\n", result.FPS)
	fmt.Fprintf(file, "最小延迟: %.3f ms\n", result.MinStableLatency)
	fmt.Fprintf(file, "最大延迟: %.3f ms\n", result.MaxStableLatency)
	fmt.Fprintf(file, "P50延迟: %.3f ms\n", result.P50StableLatency)
	fmt.Fprintf(file, "P90延迟: %.3f ms\n", result.P90StableLatency)
	fmt.Fprintf(file, "P99延迟: %.3f ms\n\n", result.P99StableLatency)

	fmt.Fprintf(file, "===== 内存使用情况 =====\n")
	fmt.Fprintf(file, "Start RSS: %.2f MB\n", result.StartRSS)
	fmt.Fprintf(file, "Cold Start RSS: %.2f MB\n", result.ColdStartRSS)
	fmt.Fprintf(file, "Stable RSS: %.2f MB\n", result.StableRSS)
	fmt.Fprintf(file, "内存增长 (Start -> Cold Start): %.2f MB\n", result.ColdStartRSS-result.StartRSS)
	fmt.Fprintf(file, "内存增长 (Cold Start -> Stable): %.2f MB\n", result.StableRSS-result.ColdStartRSS)
	fmt.Fprintf(file, "Go Heap: %.2f MB\n\n", float64(m.Alloc)/1024/1024)

	fmt.Printf("文件写入成功!\n")

	// 验证文件内容
	content, err := os.ReadFile(resultPath)
	if err != nil {
		fmt.Printf("读取文件失败: %v\n", err)
		return
	}
	fmt.Printf("\n文件内容预览:\n")
	lines := strings.Split(string(content), "\n")
	for i, line := range lines {
		if i < 50 {
			fmt.Println(line)
		}
	}

	fmt.Println("\n===== 冷启动时间对比分析测试完成 =====")
}
