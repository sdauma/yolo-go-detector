package main

import (
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"image/color"
	_ "image/jpeg"
	_ "image/png"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	modelPath = "./third_party/yolo11x.onnx"
	imagePath = "./assets/bus.jpg"
	inputSize = 640
)

type ScaleInfo struct {
	ScaleX  float32
	ScaleY  float32
	PadLeft int
	PadTop  int
}

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

func loadImage(path string) image.Image {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		panic(err)
	}
	return img
}

func fillInputTensor(img image.Image, tensor *ort.Tensor[float32]) ScaleInfo {
	data := tensor.GetData()
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	scale := float32(inputSize) / float32(max(w, h))
	nw := int(float32(w) * scale)
	nh := int(float32(h) * scale)

	resized := resizeBilinear(img, nw, nh)

	padX := (inputSize - nw) / 2
	padY := (inputSize - nh) / 2

	channelSize := inputSize * inputSize

	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			dstIdx := y*inputSize + x
			r, g, b := uint8(114), uint8(114), uint8(114)

			sx := x - padX
			sy := y - padY
			if sx >= 0 && sx < nw && sy >= 0 && sy < nh {
				cr, cg, cb, _ := resized.At(sx, sy).RGBA()
				r = uint8(cr >> 8)
				g = uint8(cg >> 8)
				b = uint8(cb >> 8)
			}

			data[dstIdx] = float32(r) / 255.0
			data[channelSize+dstIdx] = float32(g) / 255.0
			data[2*channelSize+dstIdx] = float32(b) / 255.0
		}
	}

	return ScaleInfo{
		ScaleX:  scale,
		ScaleY:  scale,
		PadLeft: padX,
		PadTop:  padY,
	}
}

func resizeBilinear(src image.Image, w, h int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	sb := src.Bounds()
	srcW := sb.Dx()
	srcH := sb.Dy()

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcX := float64(x) * float64(srcW) / float64(w)
			srcY := float64(y) * float64(srcH) / float64(h)

			x1 := int(math.Floor(srcX))
			y1 := int(math.Floor(srcY))
			x2 := min(x1+1, srcW-1)
			y2 := min(y1+1, srcH-1)

			tx := srcX - float64(x1)
			yy := srcY - float64(y1)

			c11 := src.At(x1, y1)
			c12 := src.At(x1, y2)
			c21 := src.At(x2, y1)
			c22 := src.At(x2, y2)

			r, g, b := bilinearInterpolate(c11, c12, c21, c22, tx, yy)

			dst.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return dst
}

func bilinearInterpolate(c11, c12, c21, c22 color.Color, tx, yy float64) (uint8, uint8, uint8) {
	r11, g11, b11, _ := c11.RGBA()
	r12, g12, b12, _ := c12.RGBA()
	r21, g21, b21, _ := c21.RGBA()
	r22, g22, b22, _ := c22.RGBA()

	r11, g11, b11 = r11>>8, g11>>8, b11>>8
	r12, g12, b12 = r12>>8, g12>>8, b12>>8
	r21, g21, b21 = r21>>8, g21>>8, b21>>8
	r22, g22, b22 = r22>>8, g22>>8, b22>>8

	rTop := uint8(float64(r11)*(1-tx) + float64(r21)*tx)
	gTop := uint8(float64(g11)*(1-tx) + float64(g21)*tx)
	bTop := uint8(float64(b11)*(1-tx) + float64(b21)*tx)

	rBottom := uint8(float64(r12)*(1-tx) + float64(r22)*tx)
	gBottom := uint8(float64(g12)*(1-tx) + float64(g22)*tx)
	bBottom := uint8(float64(b12)*(1-tx) + float64(b22)*tx)

	r := uint8(float64(rTop)*(1-yy) + float64(rBottom)*yy)
	g := uint8(float64(gTop)*(1-yy) + float64(gBottom)*yy)
	b := uint8(float64(bTop)*(1-yy) + float64(bBottom)*yy)

	return r, g, b
}

func main() {
	fmt.Println("===== Go 标准性能测试 (intra=1, inter=1, yolo11x) =====")
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

	// 线程配置
	opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)
	opts.SetExecutionMode(ort.ExecutionModeSequential)

	// 创建输入输出张量
	inputShape := ort.NewShape(1, 3, inputSize, inputSize)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		panic(err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		panic(err)
	}
	defer outputTensor.Destroy()

	// 保持输入数据的引用
	inputData := inputTensor.GetData()
	defer runtime.KeepAlive(inputData)

	// 加载图像并填充输入
	img := loadImage(imagePath)
	fillInputTensor(img, inputTensor)

	// 创建 Session（只创建一次）
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
	fmt.Printf("Intra-op threads: 1\n")
	fmt.Printf("Inter-op threads: 1\n")
	fmt.Printf("Execution mode: SEQUENTIAL\n")

	// 冻结 GC
	runtime.GC()
	debug.SetGCPercent(-1)

	// Warmup
	fmt.Println("\nWarming up...")
	for i := 0; i < 10; i++ {
		if err := session.Run(); err != nil {
			panic(err)
		}
		if i%2 == 0 {
			fmt.Printf("Warm-up %d/10 done\n", i+1)
		}
	}

	// Benchmark
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

	// 计算统计信息
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

	// 排序计算分位数
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

	// 打印结果
	fmt.Println("\n===== 测试结果 =====")
	fmt.Printf("Go avg: %.3f ms\n", avg)
	fmt.Printf("Go p50: %.3f ms\n", p50)
	fmt.Printf("Go p90: %.3f ms\n", p90)
	fmt.Printf("Go p99: %.3f ms\n", p99)
	fmt.Printf("Go min: %.3f ms\n", minTime)
	fmt.Printf("Go max: %.3f ms\n", maxTime)
	fmt.Printf("\nTotal runs: %d\n", N)

	// 保存结果
	file, err := os.Create("go_benchmark_std_intra1.txt")
	if err == nil {
		file.WriteString("===== Go 标准性能测试结果 (intra=1, inter=1, yolo11x) =====\n")
		file.WriteString("测试配置: warmup=10, runs=30, batch=1, concurrency=1\n")
		fmt.Fprintf(file, "Intra-op threads: 1\n")
		fmt.Fprintf(file, "Inter-op threads: 1\n")
		fmt.Fprintf(file, "Go avg: %.3f ms\n", avg)
		fmt.Fprintf(file, "Go p50: %.3f ms\n", p50)
		fmt.Fprintf(file, "Go p90: %.3f ms\n", p90)
		fmt.Fprintf(file, "Go p99: %.3f ms\n", p99)
		fmt.Fprintf(file, "Go min: %.3f ms\n", minTime)
		fmt.Fprintf(file, "Go max: %.3f ms\n", maxTime)
		fmt.Fprintf(file, "Total runs: %d\n", N)
		file.Close()
		fmt.Println("\nResults saved to go_benchmark_std_intra1.txt")
	}

	fmt.Println("\n测试完成!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
