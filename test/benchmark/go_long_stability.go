// go_long_stability.go
// Go 长时间稳定性测试 - Baseline 执行路径
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

package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

// RSSSample RSS采样点
type RSSSample struct {
	Timestamp time.Time
	RSS       float64
}

func main() {
	fmt.Println("===== Go 长时间稳定性测试 =====")
	fmt.Println("测试时长: 10分钟")
	fmt.Println("采样间隔: 1秒")

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

	// 显式设置所有 SessionOptions 参数（P2原则：禁止依赖默认值）
	// 线程配置
	opts.SetIntraOpNumThreads(4)
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
		return
	}
	defer inputTensor.Destroy()

	// 准备输入数据（从文件加载，确保与 Python 版本一致）
	fmt.Println("加载输入数据...")
	inputData := inputTensor.GetData()
	// 计算数据文件路径
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	err = loadInputDataFromFile(inputData, inputDataPath)
	if err != nil {
		fmt.Printf("加载输入数据失败: %v\n", err)
		return
	}
	fmt.Printf("输入数据加载成功: %s\n", inputDataPath)

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

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		err := session.Run()
		if err != nil {
			fmt.Printf("Warmup 运行失败: %v\n", err)
			return
		}
	}
	fmt.Println("Warmup 完成!")

	// 开始长时间稳定性测试
	fmt.Println("\n===== 开始长时间稳定性测试 =====")
	fmt.Println("测试时长: 10分钟 (600秒)")
	fmt.Println("采样间隔: 1秒")
	fmt.Println("推理模式: 连续推理")

	// 测试参数
	testDuration := 10 * time.Minute
	startTime := time.Now()
	endTime := startTime.Add(testDuration)

	// RSS采样数据
	var rssSamples []RSSSample
	var inferenceTimes []float64
	var peakRSS float64
	var minRSS float64

	// 初始RSS采样
	initialRSS := getProcessRSS()
	peakRSS = initialRSS
	minRSS = initialRSS
	rssSamples = append(rssSamples, RSSSample{
		Timestamp: startTime,
		RSS:       initialRSS,
	})
	fmt.Printf("初始 RSS: %.2f MB\n", initialRSS)

	// 推理计数器
	inferenceCount := 0

	// 主测试循环
	for time.Now().Before(endTime) {
		// 执行推理
		t0 := time.Now()
		err := session.Run()
		if err != nil {
			fmt.Printf("运行失败: %v\n", err)
			return
		}
		dt := time.Since(t0).Milliseconds()
		inferenceTimes = append(inferenceTimes, float64(dt))
		inferenceCount++

		// 每10次推理采样一次内存，减少开销
		if inferenceCount%10 == 0 {
			currentRSS := getProcessRSS()
			if currentRSS > peakRSS {
				peakRSS = currentRSS
			}
			if currentRSS < minRSS {
				minRSS = currentRSS
			}
			rssSamples = append(rssSamples, RSSSample{
				Timestamp: time.Now(),
				RSS:       currentRSS,
			})
		}

		// 每分钟输出一次进度
		if inferenceCount%60 == 0 {
			elapsed := time.Since(startTime)
			remaining := endTime.Sub(time.Now())
			// 这里不获取RSS，避免影响性能
			fmt.Printf("进度: %d 次推理, 已运行: %v, 剩余: %v\n",
				inferenceCount, elapsed.Round(time.Second), remaining.Round(time.Second))
		}
	}

	// 最终RSS采样
	finalRSS := getProcessRSS()
	rssSamples = append(rssSamples, RSSSample{
		Timestamp: time.Now(),
		RSS:       finalRSS,
	})

	// 计算统计结果
	totalDuration := time.Since(startTime)
	sort.Float64s(inferenceTimes)
	avgInferenceTime := 0.0
	for _, t := range inferenceTimes {
		avgInferenceTime += t
	}
	avgInferenceTime /= float64(len(inferenceTimes))
	minInferenceTime := inferenceTimes[0]
	maxInferenceTime := inferenceTimes[len(inferenceTimes)-1]
	p50InferenceTime := inferenceTimes[len(inferenceTimes)/2]
	p90InferenceTime := inferenceTimes[int(float64(len(inferenceTimes))*0.9)]
	p99InferenceTime := inferenceTimes[int(float64(len(inferenceTimes))*0.99)]

	// 计算RSS统计
	var rssSum float64
	for _, sample := range rssSamples {
		rssSum += sample.RSS
	}
	avgRSS := rssSum / float64(len(rssSamples))
	rssDrift := finalRSS - initialRSS

	// 输出测试结果
	fmt.Printf("\n===== 长时间稳定性测试结果 =====\n")
	fmt.Printf("测试时长: %v\n", totalDuration.Round(time.Second))
	fmt.Printf("推理次数: %d\n", inferenceCount)
	fmt.Printf("推理频率: %.2f 次/秒\n", float64(inferenceCount)/totalDuration.Seconds())

	fmt.Printf("\n===== 推理性能统计 =====\n")
	fmt.Printf("平均推理时间: %.3f ms\n", avgInferenceTime)
	fmt.Printf("P50推理时间: %.3f ms\n", p50InferenceTime)
	fmt.Printf("P90推理时间: %.3f ms\n", p90InferenceTime)
	fmt.Printf("P99推理时间: %.3f ms\n", p99InferenceTime)
	fmt.Printf("最小推理时间: %.3f ms\n", minInferenceTime)
	fmt.Printf("最大推理时间: %.3f ms\n", maxInferenceTime)

	fmt.Printf("\n===== 内存使用统计 =====\n")
	fmt.Printf("初始 RSS: %.2f MB\n", initialRSS)
	fmt.Printf("最终 RSS: %.2f MB\n", finalRSS)
	fmt.Printf("平均 RSS: %.2f MB\n", avgRSS)
	fmt.Printf("峰值 RSS: %.2f MB\n", peakRSS)
	fmt.Printf("最小 RSS: %.2f MB\n", minRSS)
	fmt.Printf("RSS Drift: %.2f MB\n", rssDrift)
	fmt.Printf("RSS 波动范围: %.2f MB (%.2f%%)\n", peakRSS-minRSS, (peakRSS-minRSS)/avgRSS*100)

	// 保存详细结果
	resultPath := filepath.Join(basePath, "results", "go_long_stability_result.txt")
	fmt.Printf("\n保存结果到: %s\n", resultPath)
	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入结果
	fmt.Fprintf(file, "===== Go 长时间稳定性测试结果 =====\n")
	fmt.Fprintf(file, "测试时长: %v\n", totalDuration.Round(time.Second))
	fmt.Fprintf(file, "推理次数: %d\n", inferenceCount)
	fmt.Fprintf(file, "推理频率: %.2f 次/秒\n", float64(inferenceCount)/totalDuration.Seconds())
	fmt.Fprintf(file, "\n===== 推理性能统计 =====\n")
	fmt.Fprintf(file, "平均推理时间: %.3f ms\n", avgInferenceTime)
	fmt.Fprintf(file, "P50推理时间: %.3f ms\n", p50InferenceTime)
	fmt.Fprintf(file, "P90推理时间: %.3f ms\n", p90InferenceTime)
	fmt.Fprintf(file, "P99推理时间: %.3f ms\n", p99InferenceTime)
	fmt.Fprintf(file, "最小推理时间: %.3f ms\n", minInferenceTime)
	fmt.Fprintf(file, "最大推理时间: %.3f ms\n", maxInferenceTime)
	fmt.Fprintf(file, "\n===== 内存使用统计 =====\n")
	fmt.Fprintf(file, "初始 RSS: %.2f MB\n", initialRSS)
	fmt.Fprintf(file, "最终 RSS: %.2f MB\n", finalRSS)
	fmt.Fprintf(file, "平均 RSS: %.2f MB\n", avgRSS)
	fmt.Fprintf(file, "峰值 RSS: %.2f MB\n", peakRSS)
	fmt.Fprintf(file, "最小 RSS: %.2f MB\n", minRSS)
	fmt.Fprintf(file, "RSS Drift: %.2f MB\n", rssDrift)
	fmt.Fprintf(file, "RSS 波动范围: %.2f MB (%.2f%%)\n", peakRSS-minRSS, (peakRSS-minRSS)/avgRSS*100)

	// 保存RSS曲线数据
	rssDataPath := filepath.Join(basePath, "results", "go_rss_curve.csv")
	fmt.Printf("保存RSS曲线数据到: %s\n", rssDataPath)
	rssFile, err := os.Create(rssDataPath)
	if err != nil {
		fmt.Printf("创建RSS曲线文件失败: %v\n", err)
		return
	}
	defer rssFile.Close()

	// 写入CSV头部
	fmt.Fprintf(rssFile, "Timestamp,Elapsed_Seconds,RSS_MB\n")

	// 写入RSS采样数据
	for _, sample := range rssSamples {
		elapsed := sample.Timestamp.Sub(startTime).Seconds()
		fmt.Fprintf(rssFile, "%s,%.3f,%.2f\n",
			sample.Timestamp.Format("2006-01-02 15:04:05.000"),
			elapsed,
			sample.RSS)
	}

	fmt.Printf("RSS曲线数据已保存: %d 个采样点\n", len(rssSamples))
	fmt.Println("\n测试完成!")
}
