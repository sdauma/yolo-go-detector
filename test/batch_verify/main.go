// Package main 批量图片检测验证工具
// 从目录读取图片，使用 SessionPool 并发推理，输出 JSONL 结果和统计报告
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	"image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/nfnt/resize"
)

var (
	imgDir    = flag.String("dir", "", "图片目录路径")
	model     = flag.String("model", "../../third_party/yolo11x.onnx", "模型路径")
	confTh    = flag.Float64("conf", 0.25, "置信度阈值")
	iouTh     = flag.Float64("iou", 0.7, "IoU 阈值")
	inSize    = flag.Int("size", 640, "输入尺寸")
	workers   = flag.Int("workers", 0, "并发数 (0=自动)")
	poolSize   = flag.Int("pool", 0, "SessionPool 大小 (0=自动)")
	limit     = flag.Int("limit", 0, "限制处理图片数 (0=全部)")
	outDir    = flag.String("out", "./output", "输出目录")
	sampleAlert = flag.Int("alert-sample", 50, "保存告警图片样本数")
	segmentSize = flag.Int("segment", 1000, "每 N 张图输出分段统计 (0=不输出)")
)

// ---------- 类型定义 ----------
type ScaleInfo struct {
	ScaleX, ScaleY float32
	PadLeft, PadTop int
}

type Detection struct {
	ClassID    int     `json:"class_id"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
	XMin       float32 `json:"xmin"`
	YMin       float32 `json:"ymin"`
	XMax       float32 `json:"xmax"`
	YMax       float32 `json:"ymax"`
}

type ImageResult struct {
	File           string      `json:"file"`
	Width          int         `json:"width"`
	Height         int         `json:"height"`
	Detections     []Detection `json:"detections"`
	InferTimeMs    float64     `json:"infer_time_ms"`
	CompletedAtSec float64     `json:"completed_at_sec"`
	Error          string      `json:"error,omitempty"`
}

type AlertSummary struct {
	Person     int `json:"person"`
	Car        int `json:"car"`
	Motorcycle int `json:"motorcycle"`
	Bus        int `json:"bus"`
	Truck      int `json:"truck"`
	Other      int `json:"other"`
	Total      int `json:"total"`
}

type RoundStats struct {
	TotalImages     int     `json:"total_images"`
	SuccessCount    int     `json:"success_count"`
	FailCount       int     `json:"fail_count"`
	TotalTimeSec    float64 `json:"total_time_sec"`
	ThroughputFPS   float64 `json:"throughput_fps"`
	AvgInferMs      float64 `json:"avg_infer_ms"`
	P50InferMs      float64 `json:"p50_infer_ms"`
	P90InferMs      float64 `json:"p90_infer_ms"`
	P99InferMs      float64 `json:"p99_infer_ms"`
	MaxInferMs      float64 `json:"max_infer_ms"`
	PeakMemoryMB    float64 `json:"peak_memory_mb"`
	TotalDetections int     `json:"total_detections"`
	AlertSummary    AlertSummary `json:"alert_summary"`
}

// TimelinePoint 每 2 秒采样的系统状态快照（用于检测渐进式退化）
type TimelinePoint struct {
	ElapsedSec float64 `json:"elapsed_sec"`
	Completed  int64   `json:"completed"`
	MemMB      float64 `json:"mem_mb"`
	Goroutines int     `json:"goroutines"`
}

// SegmentReport 每 N 张图的分段统计
type SegmentReport struct {
	Segment    int     `json:"segment"`
	From       int     `json:"from"`
	To         int     `json:"to"`
	Success    int     `json:"success"`
	Fail       int     `json:"fail"`
	FPS        float64 `json:"fps"`
	AvgInferMs float64 `json:"avg_infer_ms"`
	P50InferMs float64 `json:"p50_infer_ms"`
	P90InferMs float64 `json:"p90_infer_ms"`
	P99InferMs float64 `json:"p99_infer_ms"`
	MaxInferMs float64 `json:"max_infer_ms"`
	Detections int     `json:"detections"`
}

var (
	cocoNames []string
	onceInit  sync.Once
	ortErr    error

	alertClasses = map[int]bool{0: true, 2: true, 3: true, 5: true, 7: true} // person, car, motorcycle, bus, truck

	peakMemBytes uint64

	// 大规模稳定性测试用
	processStartTime time.Time
	totalImages      int            // 图片总数（进度打印用）
	completedCount   int64          // atomic，每张图片完成时 +1
	timeline         []TimelinePoint
	timelineMu       sync.Mutex
)

func init() {
	cocoNames = []string{
		"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
		"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
		"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
		"umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
		"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
		"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
		"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
		"couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
		"remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
		"book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
	}
}

func main() {
	flag.Parse()

	if *imgDir == "" {
		fmt.Println("用法: go run . -dir <图片目录>")
		os.Exit(1)
	}

	// 自动计算并发数：大模型（如 yolo11x）每 Session 至少需要 3-6 线程才能有效并行
	// 策略：池大小 = max(1, CPU/4)，确保 intraOp ≥ 4；上限 3。
	// 用户可通过 -pool 手动覆盖（如 -pool 2 在大模型上通常最优）
	numCPU := runtime.NumCPU()
	p := *poolSize
	if p <= 0 {
		p = max(1, numCPU/4)
		if p > 3 { p = 3 }
	}
	intraOp := numCPU / p
	if intraOp < 1 { intraOp = 1 }
	w := *workers
	if w <= 0 { w = p + 2 }
	if w > p*2 { w = p * 2 }

	// 创建输出目录
	if err := os.MkdirAll(filepath.Join(*outDir, "alerts"), 0755); err != nil {
		fmt.Printf("创建输出目录失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("批量检测验证工具\n")
	fmt.Printf("  图片目录: %s\n", *imgDir)
	fmt.Printf("  模型:     %s\n", *model)
	fmt.Printf("  参数:     conf=%.2f iou=%.2f size=%d\n", *confTh, *iouTh, *inSize)
	fmt.Printf("  并发:     workers=%d pool=%d intra_op=%d (CPU=%d核)\n", w, p, intraOp, numCPU)
	if intraOp < 3 && p > 1 {
		fmt.Printf("  💡 提示:   每 Session 仅 %d 线程，大模型可能较慢。可尝试 -pool %d\n", intraOp, max(1, numCPU/6))
	}
	fmt.Printf("  输出:     %s\n\n", *outDir)

	// 收集图片
	imgs := collectImages(*imgDir)
	if *limit > 0 && *limit < len(imgs) {
		imgs = imgs[:*limit]
	}
	fmt.Printf("  找到 %d 张图片\n\n", len(imgs))
	totalImages = len(imgs)

	if len(imgs) == 0 {
		fmt.Println("没有找到图片")
		os.Exit(1)
	}

	// 初始化 ORT
	if err := initORTEnv(); err != nil {
		fmt.Printf("初始化失败: %v\n", err)
		os.Exit(1)
	}

	// 创建 SessionPool
	sp := newSessionPool(p, *model, *inSize, intraOp)
	defer sp.destroy()

	// 处理 - Worker Pool 模式（仅 workers 个 goroutine，避免 39775 个 goroutine 的调度开销）
	results := make([]ImageResult, len(imgs))

	type task struct {
		idx  int
		path string
	}
	taskCh := make(chan task, len(imgs))
	for i, imgPath := range imgs {
		taskCh <- task{idx: i, path: imgPath}
	}
	close(taskCh)

	startTime := time.Now()
	processStartTime = startTime

	// ★ 后台时间线采样：每 2 秒记录内存、goroutine 数、完成进度
	stopTimeline := make(chan struct{})
	go collectTimeline(stopTimeline)

	var wg sync.WaitGroup
	for worker := 0; worker < w; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range taskCh {
				results[t.idx] = processImage(sp, t.path)
			}
		}()
	}
	wg.Wait()
	close(stopTimeline) // 停止时间线采样

	// 最后补一次采样
	appendTimelineSample()

	elapsed := time.Since(startTime).Seconds()

	// 统计
	stats := computeStats(results, elapsed)
	printStats(stats)
	saveResults(results, stats)

	// ★ 分段统计 + 时间线输出（大规模稳定性测试用）
	if *segmentSize > 0 && len(results) > *segmentSize {
		segments := computeSegments(results, *segmentSize)
		writeTimelineCSV()
		writeSegmentsJSON(segments)
		printSegments(segments)
	}

	fmt.Println("\n全部完成！")
}

func collectImages(dir string) []string {
	var imgs []string
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Printf("读取目录失败: %v\n", err)
		return imgs
	}
	for _, e := range entries {
		if e.IsDir() { continue }
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" {
			imgs = append(imgs, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(imgs)
	return imgs
}

// ---------- SessionPool ----------
type sessionPool struct {
	ch        chan *poolSession
	modelPath string
	inputSize int
	mu        sync.Mutex
}

type poolSession struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

func (s *poolSession) destroy() {
	if s.input != nil { s.input.Destroy() }
	if s.output != nil { s.output.Destroy() }
	if s.session != nil { s.session.Destroy() }
}

func newSessionPool(maxSize int, modelPath string, inputSize, intraOpThreads int) *sessionPool {
	sp := &sessionPool{
		ch:        make(chan *poolSession, maxSize),
		modelPath: modelPath,
		inputSize: inputSize,
	}
	// 预创建满池的 Session，确保后续 get() 只需从 channel 拿，无需创建
	for i := 0; i < maxSize; i++ {
		if s, err := createSession(modelPath, inputSize, intraOpThreads); err == nil {
			sp.ch <- s
		} else {
			fmt.Printf("  警告: 创建第 %d 个 Session 失败: %v\n", i+1, err)
		}
	}
	fmt.Printf("  SessionPool: 预创建 %d 个 Session（池容量 %d，每 Session %d 线程）\n", len(sp.ch), maxSize, intraOpThreads)
	return sp
}

func (sp *sessionPool) get() (*poolSession, error) {
	// 阻塞等待，确保不会超过池容量创建过多 Session
	// 这是关键：不阻塞会导致 12 个 worker 同时创建 12+ 个 Session，
	// CPU 争抢让每个推理从 0.92s 变成 35s+
	s := <-sp.ch
	return s, nil
}

func (sp *sessionPool) put(s *poolSession) {
	select {
	case sp.ch <- s:
	default:
		s.destroy()
	}
}

func (sp *sessionPool) destroy() {
	close(sp.ch)
	for s := range sp.ch {
		s.destroy()
	}
}

func createSession(modelPath string, inputSize, intraOpThreads int) (*poolSession, error) {
	inputShape := ort.NewShape(1, 3, int64(inputSize), int64(inputSize))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil { return nil, err }

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, err
	}
	defer opts.Destroy()

	// ★ 按池大小自动分配线程：total_threads = pool × intra_op ≤ CPU核数
	// 避免 ONNX Runtime 默认用满所有核心导致线程争抢
	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, opts)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, err
	}

	return &poolSession{session: session, input: inputTensor, output: outputTensor}, nil
}

func initORTEnv() error {
	onceInit.Do(func() {
		lib := findONNXLib()
		if lib == "" {
			ortErr = fmt.Errorf("未找到 onnxruntime.dll")
			return
		}
		ort.SetSharedLibraryPath(lib)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

func findONNXLib() string {
	for i := 0; i < 6; i++ {
		prefix := strings.Repeat("../", i)
		candidate := filepath.Join(prefix, "third_party", "onnxruntime.dll")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return "./third_party/onnxruntime.dll"
}

// ---------- 图片处理 ----------
func processImage(sp *sessionPool, imgPath string) ImageResult {
	result := ImageResult{File: filepath.Base(imgPath)}
	start := time.Now()

	f, err := os.Open(imgPath)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		result.Error = err.Error()
		return result
	}

	result.Width = img.Bounds().Dx()
	result.Height = img.Bounds().Dy()

	// 获取 session
	sess, err := sp.get()
	if err != nil {
		result.Error = err.Error()
		return result
	}
	defer sp.put(sess)

	// 预处理
	si := preprocess(img, *inSize, sess.input)
	// 推理
	if err := sess.session.Run(); err != nil {
		result.Error = err.Error()
		return result
	}
	// 后处理
	boxes := postprocess(sess.output.GetData(), result.Width, result.Height,
		float32(*confTh), float32(*iouTh), si)

	for _, b := range boxes {
		result.Detections = append(result.Detections, Detection{
			ClassID:    b.classID,
			ClassName:  cocoNames[b.classID],
			Confidence: b.conf,
			XMin:       b.x1, YMin: b.y1, XMax: b.x2, YMax: b.y2,
		})
	}

	result.InferTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	result.CompletedAtSec = time.Since(processStartTime).Seconds()

	// 更新峰值内存
	updatePeakMemory()

	// 原子递增完成计数（供时间线采样用）
	atomic.AddInt64(&completedCount, 1)

	// 保存告警图片（有检测到 person/car 等目标的）
	if *sampleAlert > 0 && hasAlert(result.Detections) {
		saveAlertSample(img, &result)
	}

	return result
}

type rawBox struct {
	classID int
	conf    float32
	x1, y1, x2, y2 float32
}

func preprocess(img image.Image, targetSize int, input *ort.Tensor[float32]) ScaleInfo {
	origW, origH := img.Bounds().Dx(), img.Bounds().Dy()
	scale := math.Min(float64(targetSize)/float64(origW), float64(targetSize)/float64(origH))
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))

	resized := resize.Resize(uint(newW), uint(newH), img, resize.Bilinear)
	// 创建 letterbox 画布
	canvas := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			canvas.Set(x, y, image.NewUniform(color.RGBA{114, 114, 114, 255}))
		}
	}
	offsetX := (targetSize - newW) / 2
	offsetY := (targetSize - newH) / 2
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			canvas.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}

	// 填充张量 (RGB 归一化到 [0,1])
	channelSize := targetSize * targetSize
	data := input.GetData()
	red, green, blue := data[:channelSize], data[channelSize:2*channelSize], data[2*channelSize:3*channelSize]
	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			r, g, b, _ := canvas.At(x, y).RGBA()
			idx := y*targetSize + x
			red[idx] = float32(r>>8) / 255.0
			green[idx] = float32(g>>8) / 255.0
			blue[idx] = float32(b>>8) / 255.0
		}
	}
	return ScaleInfo{ScaleX: float32(scale), ScaleY: float32(scale), PadLeft: offsetX, PadTop: offsetY}
}

func postprocess(output []float32, origW, origH int, confTh, iouTh float32, si ScaleInfo) []rawBox {
	numAnchors := 8400
	numClasses := 80
	boxes := make([]*rawBox, 0, 100)

	for i := 0; i < numAnchors; i++ {
		xc := output[0*numAnchors+i]
		yc := output[1*numAnchors+i]
		w := output[2*numAnchors+i]
		h := output[3*numAnchors+i]

		maxProb := float32(0)
		classID := 0
		for c := 0; c < numClasses; c++ {
			prob := output[(4+c)*numAnchors+i]
			if prob > maxProb { maxProb = prob; classID = c }
		}
		if maxProb < confTh { continue }

		ocx := (xc - float32(si.PadLeft)) / si.ScaleX
		ocy := (yc - float32(si.PadTop)) / si.ScaleY
		ow := w / si.ScaleX
		oh := h / si.ScaleY

		x1 := clamp(ocx-ow/2, 0, float32(origW))
		y1 := clamp(ocy-oh/2, 0, float32(origH))
		x2 := clamp(ocx+ow/2, 0, float32(origW))
		y2 := clamp(ocy+oh/2, 0, float32(origH))
		if x2 <= x1 || y2 <= y1 { continue }

		boxes = append(boxes, &rawBox{classID: classID, conf: maxProb, x1: x1, y1: y1, x2: x2, y2: y2})
	}

	sort.Slice(boxes, func(i, j int) bool { return boxes[i].conf > boxes[j].conf })

	// NMS per class
	selected := make([]rawBox, 0)
	picked := make([]bool, len(boxes))
	for i := range boxes {
		if picked[i] { continue }
		selected = append(selected, *boxes[i])
		picked[i] = true
		for j := i + 1; j < len(boxes); j++ {
			if picked[j] || boxes[i].classID != boxes[j].classID { continue }
			if boxIoU(boxes[i], boxes[j]) >= iouTh { picked[j] = true }
		}
	}
	return selected
}

func boxIoU(a, b *rawBox) float32 {
	xi1 := max(a.x1, b.x1)
	yi1 := max(a.y1, b.y1)
	xi2 := min(a.x2, b.x2)
	yi2 := min(a.y2, b.y2)
	inter := max(float32(0), xi2-xi1) * max(float32(0), yi2-yi1)
	areaA := (a.x2-a.x1)*(a.y2-a.y1)
	areaB := (b.x2-b.x1)*(b.y2-b.y1)
	u := areaA + areaB - inter
	if u == 0 { return 0 }
	return inter / u
}

func clamp(v, lo, hi float32) float32 {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

// ---------- 告警图片 ----------
var alertCounter int32

func hasAlert(dets []Detection) bool {
	for _, d := range dets {
		if alertClasses[d.ClassID] { return true }
	}
	return false
}

func saveAlertSample(img image.Image, result *ImageResult) {
	c := atomic.AddInt32(&alertCounter, 1)
	if c > int32(*sampleAlert) { return }

	outPath := filepath.Join(*outDir, "alerts",
		fmt.Sprintf("alert_%04d_%s", c, result.File))
	if !strings.HasSuffix(outPath, ".jpg") { outPath += ".jpg" }

	rgba := image.NewRGBA(img.Bounds())
	bounds := img.Bounds()
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	// 画框
	classColors := map[int][3]uint8{
		0: {0, 0, 255}, 2: {0, 255, 0}, 3: {255, 255, 0}, 5: {0, 255, 255}, 7: {255, 0, 0},
	}
	defCol := [3]uint8{128, 128, 128}
	for _, d := range result.Detections {
		col := classColors[d.ClassID]
		if col == [3]uint8{} { col = defCol }
		c := color.RGBA{col[0], col[1], col[2], 255}
		for y := int(d.YMin); y <= int(d.YMax); y++ {
			if y < 0 || y >= bounds.Dy() { continue }
			if int(d.XMin) >= 0 && int(d.XMin) < bounds.Dx() { rgba.Set(int(d.XMin), y, c) }
			if int(d.XMax) >= 0 && int(d.XMax) < bounds.Dx() { rgba.Set(int(d.XMax), y, c) }
		}
		for x := int(d.XMin); x <= int(d.XMax); x++ {
			if x < 0 || x >= bounds.Dx() { continue }
			if int(d.YMin) >= 0 && int(d.YMin) < bounds.Dy() { rgba.Set(x, int(d.YMin), c) }
			if int(d.YMax) >= 0 && int(d.YMax) < bounds.Dy() { rgba.Set(x, int(d.YMax), c) }
		}
	}
	f, _ := os.Create(outPath)
	if f != nil {
		defer f.Close()
		jpeg.Encode(f, rgba, &jpeg.Options{Quality: 85})
	}
}

// ---------- 统计 ----------
func updatePeakMemory() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	for {
		old := atomic.LoadUint64(&peakMemBytes)
		if m.Alloc <= old { break }
		if atomic.CompareAndSwapUint64(&peakMemBytes, old, m.Alloc) { break }
	}
}

func computeStats(results []ImageResult, elapsedSec float64) RoundStats {
	var totalInferMs float64
	var inferTimes []float64
	var success, fail int
	alert := AlertSummary{}

	for _, r := range results {
		if r.Error != "" {
			fail++
			continue
		}
		success++
		inferTimes = append(inferTimes, r.InferTimeMs)
		totalInferMs += r.InferTimeMs

		for _, d := range r.Detections {
			switch d.ClassID {
			case 0: alert.Person++
			case 2: alert.Car++
			case 3: alert.Motorcycle++
			case 5: alert.Bus++
			case 7: alert.Truck++
			default: alert.Other++
			}
			alert.Total++
		}
	}

	sort.Float64s(inferTimes)

	p50 := percentile(inferTimes, 0.5)
	p90 := percentile(inferTimes, 0.9)
	p99 := percentile(inferTimes, 0.99)
	avgInfer := totalInferMs / float64(max(len(inferTimes), 1))
	maxInfer := float64(0)
	if len(inferTimes) > 0 { maxInfer = inferTimes[len(inferTimes)-1] }

	return RoundStats{
		TotalImages:     len(results),
		SuccessCount:    success,
		FailCount:       fail,
		TotalTimeSec:    elapsedSec,
		ThroughputFPS:   float64(success) / max(elapsedSec, 0.001),
		AvgInferMs:      avgInfer,
		P50InferMs:      p50,
		P90InferMs:      p90,
		P99InferMs:      p99,
		MaxInferMs:      maxInfer,
		PeakMemoryMB:    float64(atomic.LoadUint64(&peakMemBytes)) / 1024 / 1024,
		TotalDetections: alert.Total,
		AlertSummary:    alert,
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 { return 0 }
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) { idx = len(sorted) - 1 }
	return sorted[idx]
}

func printStats(s RoundStats) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("统计报告\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))
	fmt.Printf("  总图片数:       %d\n", s.TotalImages)
	fmt.Printf("  成功:           %d\n", s.SuccessCount)
	fmt.Printf("  失败:           %d\n", s.FailCount)
	fmt.Printf("  总耗时:         %.2f 秒\n", s.TotalTimeSec)
	fmt.Printf("  吞吐量:         %.2f FPS\n", s.ThroughputFPS)
	fmt.Printf("  平均推理延迟:   %.2f ms\n", s.AvgInferMs)
	fmt.Printf("  P50 延迟:       %.2f ms\n", s.P50InferMs)
	fmt.Printf("  P90 延迟:       %.2f ms\n", s.P90InferMs)
	fmt.Printf("  P99 延迟:       %.2f ms\n", s.P99InferMs)
	fmt.Printf("  最大延迟:       %.2f ms\n", s.MaxInferMs)
	fmt.Printf("  Go 峰值内存:    %.1f MB\n", s.PeakMemoryMB)
	fmt.Printf("  总检测目标数:   %d\n", s.TotalDetections)
	fmt.Printf("    人员:         %d\n", s.AlertSummary.Person)
	fmt.Printf("    汽车:         %d\n", s.AlertSummary.Car)
	fmt.Printf("    摩托车:       %d\n", s.AlertSummary.Motorcycle)
	fmt.Printf("    巴士:         %d\n", s.AlertSummary.Bus)
	fmt.Printf("    卡车:         %d\n", s.AlertSummary.Truck)
	fmt.Printf("    其他:         %d\n", s.AlertSummary.Other)
	fmt.Printf("%s\n", strings.Repeat("=", 60))
}

// ---------- 保存结果 ----------
func saveResults(results []ImageResult, stats RoundStats) {
	// JSONL
	jsonlPath := filepath.Join(*outDir, "detections.jsonl")
	f, err := os.Create(jsonlPath)
	if err != nil {
		fmt.Printf("  保存 JSONL 失败: %v\n", err)
		return
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, r := range results {
		enc.Encode(r)
	}
	// 写入统计
	enc.Encode(stats)

	fmt.Printf("  结果已保存至: %s\n", jsonlPath)

	// 统计 JSON
	statPath := filepath.Join(*outDir, "stats.json")
	sf, err := os.Create(statPath)
	if err != nil {
		fmt.Printf("  保存统计失败: %v\n", err)
		return
	}
	defer sf.Close()
	json.NewEncoder(sf).Encode(stats)
	fmt.Printf("  统计已保存至: %s\n", statPath)
}

// ---------- 大规模稳定性测试：时间线采样 ----------

func collectTimeline(stop <-chan struct{}) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			appendTimelineSample()
		case <-stop:
			return
		}
	}
}

func appendTimelineSample() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	completed := atomic.LoadInt64(&completedCount)
	elapsed := time.Since(processStartTime).Seconds()

	pt := TimelinePoint{
		ElapsedSec: elapsed,
		Completed:  completed,
		MemMB:      float64(m.Alloc) / 1024 / 1024,
		Goroutines: runtime.NumGoroutine(),
	}
	timelineMu.Lock()
	timeline = append(timeline, pt)
	timelineMu.Unlock()

	// ★ 实时进度打印（每 2 秒一行）
	fps := float64(0)
	if elapsed > 0 {
		fps = float64(completed) / elapsed
	}
	fmt.Printf("  [%5.0fs] 已完成 %d/%d (%5.1f%%) | 内存 %7.1f MB | goroutine %d | ~%.1f FPS\n",
		elapsed, completed, totalImages, float64(completed)/float64(totalImages)*100,
		pt.MemMB, pt.Goroutines, fps)
}

// ---------- 大规模稳定性测试：分段统计 ----------

func computeSegments(results []ImageResult, segSize int) []SegmentReport {
	n := (len(results) + segSize - 1) / segSize
	segments := make([]SegmentReport, 0, n)

	for s := 0; s < n; s++ {
		from := s * segSize
		to := min(from+segSize, len(results))
		chunk := results[from:to]

		r := SegmentReport{
			Segment: s + 1,
			From:    from + 1,
			To:      to,
		}

		var inferTimes []float64
		var firstAt, lastAt float64
		firstAt = -1

		for _, img := range chunk {
			if img.Error != "" {
				r.Fail++
				continue
			}
			r.Success++
			r.Detections += len(img.Detections)
			inferTimes = append(inferTimes, img.InferTimeMs)

			if firstAt < 0 || img.CompletedAtSec < firstAt {
				firstAt = img.CompletedAtSec
			}
			if img.CompletedAtSec > lastAt {
				lastAt = img.CompletedAtSec
			}
		}

		if len(inferTimes) > 0 {
			sort.Float64s(inferTimes)

			var total float64
			for _, t := range inferTimes {
				total += t
			}
			r.AvgInferMs = total / float64(len(inferTimes))
			r.P50InferMs = percentile(inferTimes, 0.5)
			r.P90InferMs = percentile(inferTimes, 0.9)
			r.P99InferMs = percentile(inferTimes, 0.99)
			r.MaxInferMs = inferTimes[len(inferTimes)-1]

			if lastAt > firstAt && lastAt-firstAt > 0.001 {
				r.FPS = float64(len(inferTimes)) / (lastAt - firstAt)
			}
		}

		segments = append(segments, r)
	}

	return segments
}

func printSegments(segments []SegmentReport) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("分段统计报告（每 %d 张图）\n", *segmentSize)
	fmt.Printf("%s\n", strings.Repeat("=", 80))
	fmt.Printf("%-6s %-10s %-10s %-8s %-8s %-10s %-10s %-10s %-10s %-10s\n",
		"段", "范围", "成功/失败", "FPS", "检测数", "Avg(ms)", "P50(ms)", "P90(ms)", "P99(ms)", "Max(ms)")
	fmt.Printf("%s\n", strings.Repeat("-", 80))

	for _, s := range segments {
		fmt.Printf("%-6d %-10s %-10s %-8.1f %-8d %-10.1f %-10.1f %-10.1f %-10.1f %-10.1f\n",
			s.Segment,
			fmt.Sprintf("%d-%d", s.From, s.To),
			fmt.Sprintf("%d/%d", s.Success, s.Fail),
			s.FPS, s.Detections,
			s.AvgInferMs, s.P50InferMs, s.P90InferMs, s.P99InferMs, s.MaxInferMs)
	}

	// 稳定性判定
	fmt.Printf("%s\n", strings.Repeat("-", 80))
	stable := checkStability(segments)
	if stable {
		fmt.Println("  稳定性判定: ✅ 通过 — 各段 FPS/P99 波动在正常范围内，无渐进式退化")
	} else {
		fmt.Println("  稳定性判定: ⚠️ 注意 — 检测到 FPS 或 P99 波动较大，可能存在渐进式退化")
	}
	fmt.Printf("%s\n", strings.Repeat("=", 80))
}

func checkStability(segments []SegmentReport) bool {
	if len(segments) < 3 {
		return true
	}
	// 跳过第一段（预热），比较最后一段与中间的 FPS 和 P99
	mid := segments[len(segments)/2]
	last := segments[len(segments)-1]

	// FPS 下降不超过 15%
	if mid.FPS > 0 && last.FPS/mid.FPS < 0.85 {
		return false
	}
	// P99 增长不超过 50%
	if mid.P99InferMs > 0 && last.P99InferMs/mid.P99InferMs > 1.5 {
		return false
	}
	return true
}

// ---------- 大规模稳定性测试：输出文件 ----------

func writeTimelineCSV() {
	path := filepath.Join(*outDir, "timeline.csv")
	f, err := os.Create(path)
	if err != nil {
		fmt.Printf("  保存时间线 CSV 失败: %v\n", err)
		return
	}
	defer f.Close()

	f.WriteString("elapsed_sec,completed,mem_mb,goroutines\n")
	timelineMu.Lock()
	for _, p := range timeline {
		fmt.Fprintf(f, "%.1f,%d,%.1f,%d\n", p.ElapsedSec, p.Completed, p.MemMB, p.Goroutines)
	}
	timelineMu.Unlock()
	fmt.Printf("  时间线已保存至: %s (%d 个采样点)\n", path, len(timeline))
}

func writeSegmentsJSON(segments []SegmentReport) {
	path := filepath.Join(*outDir, "segments.json")
	f, err := os.Create(path)
	if err != nil {
		fmt.Printf("  保存分段统计失败: %v\n", err)
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.Encode(segments)
	fmt.Printf("  分段统计已保存至: %s (%d 段)\n", path, len(segments))
}
