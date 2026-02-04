package main

import (
	"fmt"
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

// RSSSample RSS采样点
type RSSSample struct {
	Timestamp time.Time
	RSS       float64
}

func main() {
	fmt.Println("===== Go 基准测试 =====")

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
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 创建会话选项
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("创建会话选项失败: %v\n", err)
		return
	}
	defer opts.Destroy()

	// 设置线程配置
	opts.SetIntraOpNumThreads(4)
	opts.SetInterOpNumThreads(1)

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入张量失败: %v\n", err)
		return
	}
	defer inputTensor.Destroy()

	// 准备输入数据（使用固定种子生成随机数）
	fmt.Println("生成输入数据...")
	inputData := inputTensor.GetData()
	seed := 12345
	rng := &Rand{seed: uint64(seed)}
	for i := range inputData {
		inputData[i] = rng.Float32()
	}

	// 创建会话（使用默认执行路径，绑定输入和输出张量）
	fmt.Println("创建 Session...")
	// 创建输出张量（YOLO11x 的输出形状通常为 [1, 84, 8400]）
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("创建输出张量失败: %v\n", err)
		return
	}
	defer outputTensor.Destroy()

	// 使用与 Python 相同的默认执行路径
	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		fmt.Printf("创建会话失败: %v\n", err)
		return
	}
	defer session.Destroy()
	fmt.Println("Session 创建成功!")

	// 内存采样点 1：Session 创建后、warmup 前（Start RSS）
	startRSS := getProcessRSS()
	fmt.Printf("Start RSS: %.2f MB\n", startRSS)

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		err := session.Run()
		if err != nil {
			fmt.Printf("Warmup 运行失败: %v\n", err)
			return
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
			return
		}
		dt := time.Since(t0).Seconds() * 1000.0
		sum += dt
		times[i] = dt
		fmt.Printf("Run %d: %.3f ms\n", i+1, dt)

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

	// 获取 Go heap 内存使用情况（辅助指标）
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Printf("\n===== 测试结果 =====\n")
	fmt.Printf("平均延迟: %.3f ms\n", avg_latency)
	fmt.Printf("P50延迟: %.3f ms\n", p50_latency)
	fmt.Printf("P90延迟: %.3f ms\n", p90_latency)
	fmt.Printf("P99延迟: %.3f ms\n", p99_latency)
	fmt.Printf("最小延迟: %.3f ms\n", min_latency)
	fmt.Printf("最大延迟: %.3f ms\n", max_latency)
	fmt.Printf("\n===== 内存使用情况 =====\n")
	fmt.Printf("Start RSS: %.2f MB\n", startRSS)
	fmt.Printf("Peak RSS: %.2f MB\n", peakRSS)
	fmt.Printf("Stable RSS: %.2f MB\n", stableRSS)
	fmt.Printf("RSS Drift: %.2f MB\n", stableRSS-startRSS)
	fmt.Printf("Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)

	// 保存结果
	resultPath := filepath.Join(basePath, "results", "go_baseline_result.txt")
	fmt.Printf("\n保存结果到: %s\n", resultPath)
	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入结果
	written, err := fmt.Fprintf(file, "===== Go 基准测试结果 =====\n")
	if err != nil {
		fmt.Printf("写入文件失败: %v\n", err)
		return
	}
	fmt.Printf("已写入 %d 字节\n", written)

	written, err = fmt.Fprintf(file, "平均延迟: %.3f ms\n", avg_latency)
	written, err = fmt.Fprintf(file, "P50延迟: %.3f ms\n", p50_latency)
	written, err = fmt.Fprintf(file, "P90延迟: %.3f ms\n", p90_latency)
	written, err = fmt.Fprintf(file, "P99延迟: %.3f ms\n", p99_latency)
	written, err = fmt.Fprintf(file, "最小延迟: %.3f ms\n", min_latency)
	written, err = fmt.Fprintf(file, "最大延迟: %.3f ms\n", max_latency)
	written, err = fmt.Fprintf(file, "\n===== 内存使用情况 =====\n")
	written, err = fmt.Fprintf(file, "Start RSS: %.2f MB\n", startRSS)
	written, err = fmt.Fprintf(file, "Peak RSS: %.2f MB\n", peakRSS)
	written, err = fmt.Fprintf(file, "Stable RSS: %.2f MB\n", stableRSS)
	written, err = fmt.Fprintf(file, "RSS Drift: %.2f MB\n", stableRSS-startRSS)
	written, err = fmt.Fprintf(file, "Go Heap: %.2f MB\n", float64(m.Alloc)/1024/1024)

	if err != nil {
		fmt.Printf("写入文件失败: %v\n", err)
		return
	}
	fmt.Printf("文件写入成功!\n")

	// 验证文件内容
	content, err := os.ReadFile(resultPath)
	if err != nil {
		fmt.Printf("读取文件失败: %v\n", err)
		return
	}
	fmt.Printf("文件内容:\n%s\n", string(content))
}
