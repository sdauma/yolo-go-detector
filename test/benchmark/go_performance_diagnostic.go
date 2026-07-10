// go_performance_diagnostic.go
// Go 性能诊断测试
//
// 技术说明：
// - 使用 Go baseline Session 接口（NewSession），该接口通过传入输入/输出 Tensor
//   自动启用 I/O Binding，但不接受 SessionOptions 参数
// - 线程配置由 ONNX Runtime 默认行为决定（intra_op_num_threads 默认等于 CPU 核数）
// - 代码中创建了 SessionOptions 并设置了 intraOp=12，但由于 NewSession 不接受 opts，
//   这些设置实际上不生效。保留 opts 创建代码仅用于记录意图
//
// 测试目的：
// - 验证线程配置、读取输入数据、运行 100 次推理
// - 分离测量 Tensor 构造延迟和推理延迟
// - 输出 avg/p50/p90/p99/Min/Max 等统计信息

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

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

// 性能诊断结果

type DiagnosticResult struct {
	// 线程配置
	GoMaxProcs        int
	IntraOpNumThreads int
	InterOpNumThreads int

	// 延迟统计
	InferenceLatencies          []float64
	TensorConstructionLatencies []float64
	TotalLatencies              []float64

	// 统计指标
	AvgInferenceLatency          float64
	AvgTensorConstructionLatency float64
	AvgTotalLatency              float64

	StdDevInferenceLatency float64
	P50InferenceLatency    float64
	P90InferenceLatency    float64
	P95InferenceLatency    float64

	// 内存统计
	StartRSS    float64
	PeakRSS     float64
	StableRSS   float64
	GoHeapAlloc uint64
	GoHeapSys   uint64
	GoGCs       uint32
}

// 运行诊断
func runDiagnostic() *DiagnosticResult {
	fmt.Println("===== Go 性能诊断 ======")

	// 1. 线程配置验证
	fmt.Println("\n1. 线程配置验证:")
	fmt.Printf("Go runtime GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// 2. 模型路径和输入数据
	// 获取当前工作目录
	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取工作目录失败: %v\n", err)
		return nil
	}

	// 构建项目根路径
	basePath := filepath.Dir(filepath.Dir(wd))
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	// 检查文件是否存在
	if !fileExists(modelPath) {
		fmt.Printf("模型文件不存在: %s\n", modelPath)
		return nil
	}
	if !fileExists(libPath) {
		fmt.Printf("库文件不存在: %s\n", libPath)
		return nil
	}

	// 初始化ORT
	ort.SetSharedLibraryPath(libPath)
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 3. 读取输入数据
	fmt.Println("\n2. 读取输入数据...")
	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return nil
	}
	fmt.Printf("输入数据大小: %d 字节\n", len(inputData))

	// 4. 创建会话选项
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("创建会话选项失败: %v\n", err)
		return nil
	}
	defer opts.Destroy()

	// 设置线程配置
	intraOpNumThreads := 12
	interOpNumThreads := 1
	opts.SetIntraOpNumThreads(intraOpNumThreads)
	opts.SetInterOpNumThreads(interOpNumThreads)

	fmt.Printf("设置 IntraOpNumThreads: %d\n", intraOpNumThreads)
	fmt.Printf("设置 InterOpNumThreads: %d\n", interOpNumThreads)

	// 5. 创建推理会话
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入张量失败: %v\n", err)
		return nil
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("创建输出张量失败: %v\n", err)
		return nil
	}
	defer outputTensor.Destroy()

	session, err := ort.NewSession(modelPath, []string{"images"}, []string{"output0"}, []*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		fmt.Printf("创建会话失败: %v\n", err)
		return nil
	}
	defer session.Destroy()

	// 6. 内存基准
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	startRSS := getProcessRSS()

	// 7. 预热
	fmt.Println("\n4. 预热...")
	for i := 0; i < 20; i++ {
		// 每次推理都创建新的input tensor副本，避免CPU cache效应
		inputTensorCopy, err := ort.NewEmptyTensor[float32](inputShape)
		if err != nil {
			fmt.Printf("创建输入张量失败: %v\n", err)
			return nil
		}
		// 将 []byte 转换为 []float32
		floatData := inputTensorCopy.GetData()
		for j := 0; j < len(floatData); j++ {
			if j*4 < len(inputData) {
				bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
				floatData[j] = math.Float32frombits(bits)
			}
		}

		// 推理
		if err := session.Run(); err != nil {
			fmt.Printf("推理失败: %v\n", err)
			inputTensorCopy.Destroy()
			return nil
		}
		inputTensorCopy.Destroy()
	}

	// 8. 性能测试
	fmt.Println("\n5. 性能测试...")
	const numInferences = 200

	inferenceLatencies := make([]float64, 0, numInferences)
	tensorConstructionLatencies := make([]float64, 0, numInferences)
	totalLatencies := make([]float64, 0, numInferences)

	maxRSS := startRSS

	// 创建一个固定的输入张量，避免每次都创建新的
	inputTensorCopy, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入张量失败: %v\n", err)
		return nil
	}
	defer inputTensorCopy.Destroy()

	for i := 0; i < numInferences; i++ {
		// 测量总时间
		startTime := time.Now()

		// 测量 Tensor 数据更新时间
		tensorStart := time.Now()
		// 将 []byte 转换为 []float32
		floatData := inputTensorCopy.GetData()
		for j := 0; j < len(floatData); j++ {
			if j*4 < len(inputData) {
				bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
				floatData[j] = math.Float32frombits(bits)
			}
		}
		tensorDuration := time.Since(tensorStart).Milliseconds()

		// 测量推理时间
		inferenceStart := time.Now()
		if err := session.Run(); err != nil {
			fmt.Printf("推理失败: %v\n", err)
			return nil
		}
		inferenceDuration := time.Since(inferenceStart).Milliseconds()

		totalDuration := time.Since(startTime).Milliseconds()

		// 记录延迟
		inferenceLatencies = append(inferenceLatencies, float64(inferenceDuration))
		tensorConstructionLatencies = append(tensorConstructionLatencies, float64(tensorDuration))
		totalLatencies = append(totalLatencies, float64(totalDuration))

		// 记录内存
		rss := getProcessRSS()
		if rss > maxRSS {
			maxRSS = rss
		}

		// 每10次推理打印一次进度
		if (i+1)%10 == 0 {
			fmt.Printf("完成 %d/%d 次推理\n", i+1, numInferences)
		}
	}

	// 9. 内存统计
	runtime.ReadMemStats(&memStats)
	stableRSS := getProcessRSS()

	// 10. 计算统计指标
	result := &DiagnosticResult{
		GoMaxProcs:                  runtime.GOMAXPROCS(0),
		IntraOpNumThreads:           intraOpNumThreads,
		InterOpNumThreads:           interOpNumThreads,
		InferenceLatencies:          inferenceLatencies,
		TensorConstructionLatencies: tensorConstructionLatencies,
		TotalLatencies:              totalLatencies,
		StartRSS:                    startRSS,
		PeakRSS:                     maxRSS,
		StableRSS:                   stableRSS,
		GoHeapAlloc:                 memStats.Alloc,
		GoHeapSys:                   memStats.Sys,
		GoGCs:                       memStats.NumGC,
	}

	// 计算延迟统计
	result.calculateStats()

	return result
}

// 计算统计指标
func (r *DiagnosticResult) calculateStats() {
	// 计算平均延迟
	sumInference := 0.0
	sumTensor := 0.0
	sumTotal := 0.0

	for i := 0; i < len(r.InferenceLatencies); i++ {
		sumInference += r.InferenceLatencies[i]
		sumTensor += r.TensorConstructionLatencies[i]
		sumTotal += r.TotalLatencies[i]
	}

	r.AvgInferenceLatency = sumInference / float64(len(r.InferenceLatencies))
	r.AvgTensorConstructionLatency = sumTensor / float64(len(r.TensorConstructionLatencies))
	r.AvgTotalLatency = sumTotal / float64(len(r.TotalLatencies))

	// 计算标准差
	var variance float64
	for _, latency := range r.InferenceLatencies {
		variance += math.Pow(latency-r.AvgInferenceLatency, 2)
	}
	r.StdDevInferenceLatency = math.Sqrt(variance / float64(len(r.InferenceLatencies)))

	// 计算百分位数
	sortedInference := make([]float64, len(r.InferenceLatencies))
	copy(sortedInference, r.InferenceLatencies)
	sort.Float64s(sortedInference)

	r.P50InferenceLatency = sortedInference[int(float64(len(sortedInference))*0.5)]
	r.P90InferenceLatency = sortedInference[int(float64(len(sortedInference))*0.9)]
	r.P95InferenceLatency = sortedInference[int(float64(len(sortedInference))*0.95)]
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

// 打印诊断结果
func printDiagnosticResult(result *DiagnosticResult) {
	fmt.Println("\n===== 诊断结果 =====")

	// 线程配置
	fmt.Println("\n1. 线程配置:")
	fmt.Printf("Go runtime GOMAXPROCS: %d\n", result.GoMaxProcs)
	fmt.Printf("IntraOpNumThreads: %d\n", result.IntraOpNumThreads)
	fmt.Printf("InterOpNumThreads: %d\n", result.InterOpNumThreads)

	// 延迟统计
	fmt.Println("\n2. 延迟统计:")
	fmt.Printf("平均推理延迟: %.5f ms\n", result.AvgInferenceLatency)
	fmt.Printf("平均Tensor构造延迟: %.5f ms\n", result.AvgTensorConstructionLatency)
	fmt.Printf("平均总延迟: %.5f ms\n", result.AvgTotalLatency)
	fmt.Printf("推理延迟标准差: %.5f ms\n", result.StdDevInferenceLatency)
	fmt.Printf("P50推理延迟: %.5f ms\n", result.P50InferenceLatency)
	fmt.Printf("P90推理延迟: %.5f ms\n", result.P90InferenceLatency)
	fmt.Printf("P95推理延迟: %.5f ms\n", result.P95InferenceLatency)

	// 内存统计
	fmt.Println("\n3. 内存统计:")
	fmt.Printf("Start RSS: %.2f MB\n", float64(result.StartRSS))
	fmt.Printf("Peak RSS: %.2f MB\n", float64(result.PeakRSS))
	fmt.Printf("Stable RSS: %.2f MB\n", float64(result.StableRSS))
	fmt.Printf("Go Heap Alloc: %.2f MB\n", float64(result.GoHeapAlloc)/1024/1024)
	fmt.Printf("Go Heap Sys: %.2f MB\n", float64(result.GoHeapSys)/1024/1024)
	fmt.Printf("GC 次数: %d\n", result.GoGCs)

	// 延迟分布
	fmt.Println("\n4. 延迟分布 (前20次):")
	for i := 0; i < 20 && i < len(result.InferenceLatencies); i++ {
		fmt.Printf("%2d: 推理=%.2fms, Tensor=%.2fms, 总=%.2fms\n",
			i+1, result.InferenceLatencies[i], result.TensorConstructionLatencies[i], result.TotalLatencies[i])
	}
}

func main() {
	// 确保结果目录存在
	resultsDir := "../../results"
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		fmt.Printf("创建结果目录失败: %v\n", err)
		return
	}

	// 运行诊断
	result := runDiagnostic()
	if result == nil {
		fmt.Println("诊断失败")
		return
	}

	// 打印结果
	printDiagnosticResult(result)

	// 保存结果
	resultPath := filepath.Join(resultsDir, "go_performance_diagnostic_result.txt")
	file, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("创建结果文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入详细结果
	fmt.Fprintln(file, "===== Go 性能诊断详细结果 =====")
	fmt.Fprintf(file, "Go runtime GOMAXPROCS: %d\n", result.GoMaxProcs)
	fmt.Fprintf(file, "IntraOpNumThreads: %d\n", result.IntraOpNumThreads)
	fmt.Fprintf(file, "InterOpNumThreads: %d\n", result.InterOpNumThreads)
	fmt.Fprintf(file, "平均推理延迟: %.5f ms\n", result.AvgInferenceLatency)
	fmt.Fprintf(file, "平均Tensor构造延迟: %.5f ms\n", result.AvgTensorConstructionLatency)
	fmt.Fprintf(file, "平均总延迟: %.5f ms\n", result.AvgTotalLatency)
	fmt.Fprintf(file, "推理延迟标准差: %.5f ms\n", result.StdDevInferenceLatency)
	fmt.Fprintf(file, "P50推理延迟: %.5f ms\n", result.P50InferenceLatency)
	fmt.Fprintf(file, "P90推理延迟: %.5f ms\n", result.P90InferenceLatency)
	fmt.Fprintf(file, "P95推理延迟: %.5f ms\n", result.P95InferenceLatency)
	fmt.Fprintf(file, "Start RSS: %.2f MB\n", float64(result.StartRSS))
	fmt.Fprintf(file, "Peak RSS: %.2f MB\n", float64(result.PeakRSS))
	fmt.Fprintf(file, "Stable RSS: %.2f MB\n", float64(result.StableRSS))
	fmt.Fprintf(file, "Go Heap Alloc: %.2f MB\n", float64(result.GoHeapAlloc)/1024/1024)
	fmt.Fprintf(file, "Go Heap Sys: %.2f MB\n", float64(result.GoHeapSys)/1024/1024)
	fmt.Fprintf(file, "GC 次数: %d\n", result.GoGCs)

	// 写入延迟数据
	fmt.Fprintln(file, "\n===== 详细延迟数据 =====")
	fmt.Fprintln(file, "序号,推理延迟(ms),Tensor构造延迟(ms),总延迟(ms)")
	for i := 0; i < len(result.InferenceLatencies); i++ {
		fmt.Fprintf(file, "%d,%.5f,%.5f,%.5f\n",
			i+1, result.InferenceLatencies[i], result.TensorConstructionLatencies[i], result.TotalLatencies[i])
	}

	fmt.Printf("\n详细结果已保存到: %s\n", resultPath)
	fmt.Println("\n诊断完成!")
}