// go_session_creation_benchmark.go
// Go Session 创建时间测试
//
// 技术说明：
// - 使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
//   自动启用 I/O Binding，但不接受 SessionOptions 参数
// - 线程配置由 ONNX Runtime 默认行为决定，未显式设置
//
// 测试目的：
// - 分别对 YOLO11x 和 YOLO11n 模型各创建 100 次 Session，测量每次创建耗时
// - 统计 avg/std/p50/p90/min/max 等指标
// - 与 Python 的 Session 创建时间进行对比

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type SessionCreationResult struct {
	AvgTime float64
	StdTime float64
	P50Time float64
	P90Time float64
	MinTime float64
	MaxTime float64
	Times   []float64
}

func runSessionCreationBenchmark(modelName, modelPath string) SessionCreationResult {
	fmt.Printf("===== Go Session创建时间测试 - %s =====\n", modelName)
	fmt.Println("CPU核心调度：系统默认")

	// 测试Session创建时间
	fmt.Printf("测试%s模型的Session创建时间...\n", modelName)
	runs := 100 // 创建100次Session
	times := make([]float64, runs)

	for i := 0; i < runs; i++ {
		start := time.Now()
		opts, err := ort.NewSessionOptions()
		if err != nil {
			fmt.Printf("错误: 创建 SessionOptions 失败: %v\n", err)
			os.Exit(1)
		}

		// 创建输入张量
		inputShape := ort.NewShape(1, 3, 640, 640)
		inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
		if err != nil {
			fmt.Printf("错误: 创建输入张量失败: %v\n", err)
			opts.Destroy()
			os.Exit(1)
		}

		// 创建输出张量
		outputShape := ort.NewShape(1, 84, 8400)
		outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			fmt.Printf("错误: 创建输出张量失败: %v\n", err)
			opts.Destroy()
			inputTensor.Destroy()
			os.Exit(1)
		}

		// 使用正确的 API 创建会话
		sess, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
		if err != nil {
			fmt.Printf("错误: 创建 InferenceSession 失败: %v\n", err)
			opts.Destroy()
			inputTensor.Destroy()
			outputTensor.Destroy()
			os.Exit(1)
		}
		elapsed := time.Since(start).Milliseconds()
		times[i] = float64(elapsed)

		// 手动释放资源（不能在循环中使用defer）
		sess.Destroy()
		opts.Destroy()
		inputTensor.Destroy()
		outputTensor.Destroy()
	}

	// 计算结果
	sum := 0.0
	for _, t := range times {
		sum += t
	}
	avgTime := sum / float64(len(times))

	// 计算标准差
	variance := 0.0
	for _, t := range times {
		diff := t - avgTime
		variance += diff * diff
	}
	stdTime := variance / float64(len(times))
	if stdTime > 0 {
		stdTime = stdTime
	}

	minTime := times[0]
	maxTime := times[0]
	for _, t := range times {
		if t < minTime {
			minTime = t
		}
		if t > maxTime {
			maxTime = t
		}
	}

	// 计算百分位
	sortedTimes := make([]float64, len(times))
	copy(sortedTimes, times)
	sort.Float64s(sortedTimes)

	p50Index := len(sortedTimes) * 50 / 100
	p90Index := len(sortedTimes) * 90 / 100

	p50Time := sortedTimes[p50Index]
	p90Time := sortedTimes[p90Index]

	return SessionCreationResult{
		AvgTime: avgTime,
		StdTime: stdTime,
		P50Time: p50Time,
		P90Time: p90Time,
		MinTime: minTime,
		MaxTime: maxTime,
		Times:   times,
	}
}

func main() {
	fmt.Println("===== Go Session创建时间测试 ====")
	fmt.Println("测试配置：")
	fmt.Println("- 线程数: 12 (默认)")
	fmt.Println("- 创建次数: 100次")
	fmt.Println()

	// 获取项目根路径
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(currentDir))

	// 初始化 ONNX Runtime
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 测试 YOLO11x 模型
	fmt.Println("===== 测试 YOLO11x 模型 =====")
	yolo11xPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	if _, err := os.Stat(yolo11xPath); os.IsNotExist(err) {
		fmt.Printf("错误: YOLO11x模型文件不存在: %s\n", yolo11xPath)
		os.Exit(1)
	}

	yolo11xResult := runSessionCreationBenchmark("YOLO11x", yolo11xPath)

	fmt.Printf("\nYOLO11x Session创建时间结果:\n")
	fmt.Printf("平均时间: %.5f ms\n", yolo11xResult.AvgTime)
	fmt.Printf("标准差: %.5f ms\n", yolo11xResult.StdTime)
	fmt.Printf("P50时间: %.5f ms\n", yolo11xResult.P50Time)
	fmt.Printf("P90时间: %.5f ms\n", yolo11xResult.P90Time)
	fmt.Printf("最小时间: %.5f ms\n", yolo11xResult.MinTime)
	fmt.Printf("最大时间: %.5f ms\n", yolo11xResult.MaxTime)

	// 测试 YOLO11n 模型
	fmt.Println("\n===== 测试 YOLO11n 模型 =====")
	yolo11nPath := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	if _, err := os.Stat(yolo11nPath); os.IsNotExist(err) {
		fmt.Printf("错误: YOLO11n模型文件不存在: %s\n", yolo11nPath)
		os.Exit(1)
	}

	yolo11nResult := runSessionCreationBenchmark("YOLO11n", yolo11nPath)

	fmt.Printf("\nYOLO11n Session创建时间结果:\n")
	fmt.Printf("平均时间: %.5f ms\n", yolo11nResult.AvgTime)
	fmt.Printf("标准差: %.5f ms\n", yolo11nResult.StdTime)
	fmt.Printf("P50时间: %.5f ms\n", yolo11nResult.P50Time)
	fmt.Printf("P90时间: %.5f ms\n", yolo11nResult.P90Time)
	fmt.Printf("最小时间: %.5f ms\n", yolo11nResult.MinTime)
	fmt.Printf("最大时间: %.5f ms\n", yolo11nResult.MaxTime)

	// 保存结果
	resultsDir := filepath.Join(basePath, "results")
	os.MkdirAll(resultsDir, 0755)

	resultPath := filepath.Join(resultsDir, "go_session_creation_result.txt")
	resultFile, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer resultFile.Close()

	// 获取系统信息
	fmt.Fprintf(resultFile, "===== Go Session创建时间测试结果 =====\n")
	fmt.Fprintf(resultFile, "测试时间: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(resultFile, "Go版本: %s\n", "1.23.2")
	fmt.Fprintf(resultFile, "测试配置：\n")
	fmt.Fprintf(resultFile, "- 线程数: 12 (默认)\n")
	fmt.Fprintf(resultFile, "- 创建次数: 100次\n")
	fmt.Fprintf(resultFile, "\n")

	fmt.Fprintf(resultFile, "===== YOLO11x 测试结果 =====\n")
	fmt.Fprintf(resultFile, "模型: YOLO11x\n")
	fmt.Fprintf(resultFile, "平均时间: %.5f ms\n", yolo11xResult.AvgTime)
	fmt.Fprintf(resultFile, "标准差: %.5f ms\n", yolo11xResult.StdTime)
	fmt.Fprintf(resultFile, "P50时间: %.5f ms\n", yolo11xResult.P50Time)
	fmt.Fprintf(resultFile, "P90时间: %.5f ms\n", yolo11xResult.P90Time)
	fmt.Fprintf(resultFile, "最小时间: %.5f ms\n", yolo11xResult.MinTime)
	fmt.Fprintf(resultFile, "最大时间: %.5f ms\n", yolo11xResult.MaxTime)
	fmt.Fprintf(resultFile, "\n")

	fmt.Fprintf(resultFile, "===== YOLO11n 测试结果 =====\n")
	fmt.Fprintf(resultFile, "模型: YOLO11n\n")
	fmt.Fprintf(resultFile, "平均时间: %.5f ms\n", yolo11nResult.AvgTime)
	fmt.Fprintf(resultFile, "标准差: %.5f ms\n", yolo11nResult.StdTime)
	fmt.Fprintf(resultFile, "P50时间: %.5f ms\n", yolo11nResult.P50Time)
	fmt.Fprintf(resultFile, "P90时间: %.5f ms\n", yolo11nResult.P90Time)
	fmt.Fprintf(resultFile, "最小时间: %.5f ms\n", yolo11nResult.MinTime)
	fmt.Fprintf(resultFile, "最大时间: %.5f ms\n", yolo11nResult.MaxTime)

	fmt.Printf("\n结果已保存到: %s\n", resultPath)
	fmt.Println("测试完成!")
}
