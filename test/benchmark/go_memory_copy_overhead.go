package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type MemoryCopyResult struct {
	DataCopyTime    float64
	CGOCallTime     float64
	GCPauseTime     float64
	TotalOverhead   float64
	InferenceTime   float64
	OverheadPercent float64
}

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

func runMemoryCopyBenchmark(session *ort.Session, inputData []byte, inputShape []int64) *MemoryCopyResult {
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

	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return
	}

	runtime.GOMAXPROCS(12)

	env := ort.NewEnvironment(ort.LogLevel(ort.ORT_LOGGING_LEVEL_WARNING))
	defer env.Destroy()

	session, err := ort.NewAdvancedSession(modelPath, []string{"images"}, []string{"output0"}, nil)
	if err != nil {
		fmt.Printf("创建Session失败: %v\n", err)
		return
	}
	defer session.Destroy()

	inputShape := []int64{1, 3, 640, 640}

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

	fmt.Printf("初始RSS: %.5f MB\n", startRSS)
	fmt.Printf("最终RSS: %.5f MB\n", endRSS)
	fmt.Printf("RSS漂移: %.5f MB\n", rssDrift)

	resultPath := filepath.Join(basePath, "results", "go_memory_copy_overhead_result.txt")
	os.MkdirAll(filepath.Dir(resultPath), 0755)

	resultContent := fmt.Sprintf("===== Go 内存拷贝和线程调度开销测试结果 =====\n\n")
	resultContent += fmt.Sprintf("数据拷贝时间: %.5f ms\n", avgDataCopy)
	resultContent += fmt.Sprintf("CGO调用开销: %.5f ms\n", avgCGOCall)
	resultContent += fmt.Sprintf("GC暂停时间: %.5f ms\n", avgGCPause)
	resultContent += fmt.Sprintf("总开销时间: %.5f ms\n", avgTotalOverhead)
	resultContent += fmt.Sprintf("推理时间: %.5f ms\n", avgInference)
	resultContent += fmt.Sprintf("开销占比: %.5f%%\n\n", avgOverheadPercent)
	resultContent += fmt.Sprintf("初始RSS: %.5f MB\n", startRSS)
	resultContent += fmt.Sprintf("最终RSS: %.5f MB\n", endRSS)
	resultContent += fmt.Sprintf("RSS漂移: %.5f MB\n", rssDrift)

	os.WriteFile(resultPath, []byte(resultContent), 0644)
	fmt.Printf("\n结果已保存到: %s\n", resultPath)
}
