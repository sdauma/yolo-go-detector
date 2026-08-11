// go_memory_copy_overhead.go
// Go 内存拷贝开销分析测试
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 nil 作为 opts（无显式线程配置）
// - 传入 nil 作为输入/输出 Tensor（不启用 I/O Binding）
// - 使用 ort.NewEnvironment 创建环境，不设置线程数
//
// 测试目的：
// - 测量每次推理中 Data Copy 时间、CGO Call 时间、GC Pause 时间相对于纯推理时间的占比
// - 共 10 轮，计算平均开销百分比
// - 分析 Go ↔ ONNX Runtime 之间的数据传输开销

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

type MemoryCopyResult struct {
	DataCopyTime    float64
	CGOCallTime     float64
	GCPauseTime     float64
	TotalOverhead   float64
	InferenceTime   float64
	OverheadPercent float64
}

// getProcessRSS returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getProcessRSS() float64 { return memutil.PrivateMemoryMB() }

func measureDataCopyOverhead(inputData []byte, inputShape []int64) (float64, float64) {
	startTime := time.Now()

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return 0, 0
	}
	defer inputTensor.Destroy()

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	dataCopyTime := time.Since(startTime).Milliseconds()

	cgoStartTime := time.Now()
	for i := 0; i < 100; i++ {
		inputTensorCopy, _ := ort.NewEmptyTensor[float32](inputShape)
		if inputTensorCopy != nil {
			inputTensorCopy.Destroy()
		}
	}
	cgoCallTime := float64(time.Since(cgoStartTime).Milliseconds()) / 100.0

	return float64(dataCopyTime), cgoCallTime
}

func measureGCPause() float64 {
	runtime.GC()
	startTime := time.Now()
	runtime.GC()
	gcPauseTime := float64(time.Since(startTime).Milliseconds())
	return gcPauseTime
}

func runMemoryCopyBenchmark(session *ort.AdvancedSession, inputData []byte, inputShape []int64) *MemoryCopyResult {
	dataCopyTime, cgoCallTime := measureDataCopyOverhead(inputData, inputShape)
	gcPauseTime := measureGCPause()

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil
	}
	defer inputTensor.Destroy()

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	startTime := time.Now()
	if err := session.Run(); err != nil {
		return nil
	}
	inferenceTime := float64(time.Since(startTime).Milliseconds())

	totalOverhead := dataCopyTime + cgoCallTime + gcPauseTime
	overheadPercent := (totalOverhead / inferenceTime) * 100

	return &MemoryCopyResult{
		DataCopyTime:    dataCopyTime,
		CGOCallTime:     cgoCallTime,
		GCPauseTime:     gcPauseTime,
		TotalOverhead:   totalOverhead,
		InferenceTime:   inferenceTime,
		OverheadPercent: overheadPercent,
	}
}

func main() {
	fmt.Println("===== Go 内存拷贝和线程调度开销测试 =====")

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	ort.SetSharedLibraryPath(filepath.Join(basePath, "third_party", "onnxruntime-win-x64-1.23.2", "lib", "onnxruntime.dll"))
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("Failed to initialize ONNX Runtime: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return
	}

	runtime.GOMAXPROCS(12)

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("创建SessionOptions失败: %v\n", err)
		return
	}
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(1)

	it, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		fmt.Printf("创建输入Tensor失败: %v\n", err)
		return
	}
	defer it.Destroy()

	ot, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("创建输出Tensor失败: %v\n", err)
		return
	}
	defer ot.Destroy()

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{it}, []ort.ArbitraryTensor{ot}, opts)
	if err != nil {
		fmt.Printf("创建Session失败: %v\n", err)
		return
	}
	defer session.Destroy()

	fmt.Println("\n===== 内存拷贝开销分析 =====")
	results := make([]*MemoryCopyResult, 0, 10)

	for i := 0; i < 10; i++ {
		result := runMemoryCopyBenchmark(session, inputData, inputShape)
		if result != nil {
			results = append(results, result)
		}
	}

	avgDataCopy := 0.0
	avgCGOCall := 0.0
	avgGCPause := 0.0
	avgTotalOverhead := 0.0
	avgInference := 0.0
	avgOverheadPercent := 0.0

	for _, r := range results {
		avgDataCopy += r.DataCopyTime
		avgCGOCall += r.CGOCallTime
		avgGCPause += r.GCPauseTime
		avgTotalOverhead += r.TotalOverhead
		avgInference += r.InferenceTime
		avgOverheadPercent += r.OverheadPercent
	}

	n := float64(len(results))
	avgDataCopy /= n
	avgCGOCall /= n
	avgGCPause /= n
	avgTotalOverhead /= n
	avgInference /= n
	avgOverheadPercent /= n

	fmt.Printf("数据拷贝时间: %.5f ms\n", avgDataCopy)
	fmt.Printf("CGO调用开销: %.5f ms\n", avgCGOCall)
	fmt.Printf("GC暂停时间: %.5f ms\n", avgGCPause)
	fmt.Printf("总开销时间: %.5f ms\n", avgTotalOverhead)
	fmt.Printf("推理时间: %.5f ms\n", avgInference)
	fmt.Printf("开销占比: %.5f%%\n", avgOverheadPercent)

	fmt.Println("\n===== 线程调度开销测试 =====")
	threadCounts := []int{1, 2, 4, 8, 12}

	for _, threadCount := range threadCounts {
		fmt.Printf("\n测试 %d 线程配置...\n", threadCount)

		var wg sync.WaitGroup
		times := make([]float64, 0, 100)

		for i := 0; i < 100; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				startTime := time.Now()
				runtime.Gosched()
				elapsed := float64(time.Since(startTime).Nanoseconds()) / 1000000.0
				times = append(times, elapsed)
			}()
		}

		wg.Wait()

		avgTime := 0.0
		for _, t := range times {
			avgTime += t
		}
		avgTime /= float64(len(times))

		fmt.Printf("平均线程调度时间: %.5f ms\n", avgTime)
	}

	fmt.Println("\n===== 内存使用分析 =====")
	startRSS := getProcessRSS()

	for i := 0; i < 100; i++ {
		inputTensor, _ := ort.NewEmptyTensor[float32](inputShape)
		if inputTensor != nil {
			inputTensor.Destroy()
		}
		runtime.GC()
	}

	endRSS := getProcessRSS()
	rssDrift := endRSS - startRSS

	fmt.Printf("初始PM: %.5f MB\n", startRSS)
	fmt.Printf("最终RSS: %.5f MB\n", endRSS)
	fmt.Printf("PM漂移: %.5f MB\n", rssDrift)

	resultPath := filepath.Join(basePath, "results", "go_memory_copy_overhead_result.txt")
	os.MkdirAll(filepath.Dir(resultPath), 0755)

	resultContent := fmt.Sprintf("===== Go 内存拷贝和线程调度开销测试结果 =====\n\n")
	resultContent += fmt.Sprintf("数据拷贝时间: %.5f ms\n", avgDataCopy)
	resultContent += fmt.Sprintf("CGO调用开销: %.5f ms\n", avgCGOCall)
	resultContent += fmt.Sprintf("GC暂停时间: %.5f ms\n", avgGCPause)
	resultContent += fmt.Sprintf("总开销时间: %.5f ms\n", avgTotalOverhead)
	resultContent += fmt.Sprintf("推理时间: %.5f ms\n", avgInference)
	resultContent += fmt.Sprintf("开销占比: %.5f%%\n\n", avgOverheadPercent)
	resultContent += fmt.Sprintf("初始PM: %.5f MB\n", startRSS)
	resultContent += fmt.Sprintf("最终RSS: %.5f MB\n", endRSS)
	resultContent += fmt.Sprintf("PM漂移: %.5f MB\n", rssDrift)

	os.WriteFile(resultPath, []byte(resultContent), 0644)
	fmt.Printf("\n结果已保存到: %s\n", resultPath)
}