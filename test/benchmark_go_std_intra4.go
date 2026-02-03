package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	modelPath = "./third_party/yolo11x.onnx"
	inputSize = 640
)

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	return ""
}

func main() {
	fmt.Println("===== Go 标准性能测试 (intra=4, inter=1, yolo11x) =====")
	fmt.Println("测试配置: warmup=10, runs=30, batch=1, concurrency=1")

	libPath := getSharedLibPath()
	if libPath == "" {
		panic("未找到ONNX Runtime库")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		panic(fmt.Errorf("初始化ORT环境失败: %w", err))
	}
	defer ort.DestroyEnvironment()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		panic(err)
	}
	defer opts.Destroy()

	opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	opts.SetIntraOpNumThreads(4)
	opts.SetInterOpNumThreads(1)
	opts.SetExecutionMode(ort.ExecutionModeSequential)

	fmt.Println("创建输入张量...")
	inputShape := ort.NewShape(1, 3, inputSize, inputSize)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		panic(err)
	}
	defer inputTensor.Destroy()

	fmt.Println("创建输出张量...")
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		panic(err)
	}
	defer outputTensor.Destroy()

	inputData := inputTensor.GetData()
	defer runtime.KeepAlive(inputData)

	fmt.Println("创建会话...")
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		opts,
	)
	if err != nil {
		panic(err)
	}
	defer session.Destroy()

	fmt.Printf("Input shape: %v\n", inputShape)
	fmt.Printf("Output shape: %v\n", outputShape)
	fmt.Printf("Intra-op threads: 4\n")
	fmt.Printf("Inter-op threads: 1\n")

	runtime.GC()
	debug.SetGCPercent(-1)

	fmt.Println("\nWarming up...")
	for i := 0; i < 10; i++ {
		if err := session.Run(); err != nil {
			panic(err)
		}
		if i%2 == 0 {
			fmt.Printf("Warm-up %d/10 done\n", i+1)
		}
	}

	fmt.Println("\nRunning benchmark...")
	const N = 30
	times := make([]float64, 0, N)

	for i := 0; i < N; i++ {
		if i%5 == 0 {
			fmt.Printf("Progress: %d/%d\n", i, N)
		}

		t0 := time.Now()
		if err := session.Run(); err != nil {
			panic(err)
		}
		dt := time.Since(t0).Seconds() * 1000.0
		times = append(times, dt)
	}

	var sum float64
	minTime := times[0]
	maxTime := times[0]

	for _, t := range times {
		sum += t
		if t < minTime {
			minTime = t
		}
		if t > maxTime {
			maxTime = t
		}
	}

	avg := sum / float64(N)

	for i := 0; i < len(times)-1; i++ {
		for j := 0; j < len(times)-i-1; j++ {
			if times[j] > times[j+1] {
				times[j], times[j+1] = times[j+1], times[j]
			}
		}
	}

	p50 := times[N/2]
	p90 := times[int(math.Floor(float64(N)*0.9))]
	p99 := times[int(math.Floor(float64(N)*0.99))]

	fmt.Println("\n===== 测试结果 =====")
	fmt.Printf("Go avg: %.3f ms\n", avg)
	fmt.Printf("Go p50: %.3f ms\n", p50)
	fmt.Printf("Go p90: %.3f ms\n", p90)
	fmt.Printf("Go p99: %.3f ms\n", p99)
	fmt.Printf("Go min: %.3f ms\n", minTime)
	fmt.Printf("Go max: %.3f ms\n", maxTime)
	fmt.Printf("\nTotal runs: %d\n", N)

	file, err := os.Create("go_benchmark_std_intra4.txt")
	if err == nil {
		file.WriteString("===== Go 标准性能测试结果 (intra=4, inter=1, yolo11x) =====\n")
		file.WriteString("测试配置: warmup=10, runs=30, batch=1, concurrency=1\n")
		fmt.Fprintf(file, "Intra-op threads: 4\n")
		fmt.Fprintf(file, "Inter-op threads: 1\n")
		fmt.Fprintf(file, "Go avg: %.3f ms\n", avg)
		fmt.Fprintf(file, "Go p50: %.3f ms\n", p50)
		fmt.Fprintf(file, "Go p90: %.3f ms\n", p90)
		fmt.Fprintf(file, "Go p99: %.3f ms\n", p99)
		fmt.Fprintf(file, "Go min: %.3f ms\n", minTime)
		fmt.Fprintf(file, "Go max: %.3f ms\n", maxTime)
		fmt.Fprintf(file, "Total runs: %d\n", N)
		file.Close()
		fmt.Println("\nResults saved to go_benchmark_std_intra4.txt")
	}

	fmt.Println("\n测试完成!")
}
