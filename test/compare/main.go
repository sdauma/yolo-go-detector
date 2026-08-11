// Package main 实现 Python vs Go 检测结果一致性对比
// 输入图片，输出 YOLO 格式 txt 标注文件供 Python 对比
package main

import (
	"errors"
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
	"sort"
	"strings"
	"sync"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	modelPath  = flag.String("model", "../../third_party/yolo11x.onnx", "模型路径")
	imgPath    = flag.String("img", "../../assets/bus.jpg", "输入图像路径")
	confThresh = flag.Float64("conf", 0.25, "置信度阈值")
	iouThresh  = flag.Float64("iou", 0.7, "IoU 阈值")
	inputSize  = flag.Int("size", 640, "输入尺寸")
	outTxt     = flag.String("out-txt", "./bus_go_detections.txt", "输出 txt 路径（YOLO 格式）")
	outImg     = flag.String("out-img", "./bus_go_result.jpg", "输出标注图片路径")

	ortInitOnce sync.Once
)

type BBox struct {
	Label          string
	ClassID        int
	Confidence     float32
	X1, Y1, X2, Y2 float32
}

type ScaleInfo struct {
	ScaleX, ScaleY  float32
	PadLeft, PadTop int
}

// ---------- ONNX Runtime 初始化 ----------
func initORT() error {
	var initErr error
	ortInitOnce.Do(func() {
		libPath := findLib()
		if libPath == "" {
			initErr = errors.New("未找到 onnxruntime.dll")
			return
		}
		ort.SetSharedLibraryPath(libPath)
		if e := ort.InitializeEnvironment(); e != nil {
			initErr = fmt.Errorf("初始化 ORT 环境失败: %w", e)
		}
	})
	return initErr
}

func findLib() string {
	for i := 0; i < 6; i++ {
		prefix := strings.Repeat("../", i)
		candidate := filepath.Join(prefix, "third_party", "onnxruntime.dll")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return ""
}

// ---------- 推理 ----------
func infer(img image.Image) ([]BBox, ScaleInfo, error) {
	if err := initORT(); err != nil {
		return nil, ScaleInfo{}, err
	}

	size := *inputSize
	inputShape := ort.NewShape(1, 3, int64(size), int64(size))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, ScaleInfo{}, err
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, ScaleInfo{}, err
	}
	defer outputTensor.Destroy()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, ScaleInfo{}, err
	}
	defer opts.Destroy()

	session, err := ort.NewAdvancedSession(*modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, opts)
	if err != nil {
		return nil, ScaleInfo{}, fmt.Errorf("创建会话失败: %w", err)
	}
	defer session.Destroy()

	// 预处理
	resized, si := letterbox(img, size)
	channelSize := size * size
	data := inputTensor.GetData()
	red := data[:channelSize]
	green := data[channelSize : 2*channelSize]
	blue := data[2*channelSize : 3*channelSize]
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*size + x
			red[idx] = float32(r>>8) / 255.0
			green[idx] = float32(g>>8) / 255.0
			blue[idx] = float32(b>>8) / 255.0
		}
	}

	if err := session.Run(); err != nil {
		return nil, ScaleInfo{}, fmt.Errorf("推理失败: %w", err)
	}

	boxes := processOutput(outputTensor.GetData(), img.Bounds().Dx(), img.Bounds().Dy(),
		float32(*confThresh), float32(*iouThresh), si)

	return boxes, si, nil
}

func letterbox(img image.Image, targetSize int) (image.Image, ScaleInfo) {
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	scale := math.Min(float64(targetSize)/float64(origW), float64(targetSize)/float64(origH))
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))

	resized := resize.Resize(uint(newW), uint(newH), img, resize.Bilinear)
	result := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))

	offsetX := (targetSize - newW) / 2
	offsetY := (targetSize - newH) / 2

	// 灰色填充
	gray := color.RGBA{114, 114, 114, 255}
	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			result.Set(x, y, gray)
		}
	}
	// 居中绘制
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			result.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}

	return result, ScaleInfo{ScaleX: float32(scale), ScaleY: float32(scale), PadLeft: offsetX, PadTop: offsetY}
}

func processOutput(output []float32, origW, origH int, confTh, iouTh float32, si ScaleInfo) []BBox {
	numAnchors := 8400
	numClasses := 80
	boxes := make([]*BBox, 0, 100)

	for i := 0; i < numAnchors; i++ {
		xc := output[0*numAnchors+i]
		yc := output[1*numAnchors+i]
		w := output[2*numAnchors+i]
		h := output[3*numAnchors+i]

		maxProb := float32(0)
		classID := 0
		for c := 0; c < numClasses; c++ {
			prob := output[(4+c)*numAnchors+i]
			if prob > maxProb {
				maxProb = prob
				classID = c
			}
		}
		if maxProb < confTh {
			continue
		}

		origCX := (xc - float32(si.PadLeft)) / si.ScaleX
		origCY := (yc - float32(si.PadTop)) / si.ScaleY
		boxW := w / si.ScaleX
		boxH := h / si.ScaleY

		x1 := origCX - boxW/2
		y1 := origCY - boxH/2
		x2 := origCX + boxW/2
		y2 := origCY + boxH/2

		x1 = clamp(x1, 0, float32(origW))
		y1 = clamp(y1, 0, float32(origH))
		x2 = clamp(x2, 0, float32(origW))
		y2 = clamp(y2, 0, float32(origH))

		if x2 <= x1 || y2 <= y1 {
			continue
		}

		boxes = append(boxes, &BBox{
			Label:      cocoNames[classID],
			ClassID:    classID,
			Confidence: maxProb,
			X1:         x1, Y1: y1, X2: x2, Y2: y2,
		})
	}

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Confidence > boxes[j].Confidence
	})

	// NMS per class
	selected := make([]BBox, 0)
	picked := make([]bool, len(boxes))
	for i := range boxes {
		if picked[i] {
			continue
		}
		selected = append(selected, *boxes[i])
		picked[i] = true
		for j := i + 1; j < len(boxes); j++ {
			if picked[j] || boxes[i].ClassID != boxes[j].ClassID {
				continue
			}
			if iou(boxes[i], boxes[j]) >= iouTh {
				picked[j] = true
			}
		}
	}
	return selected
}

func iou(a, b *BBox) float32 {
	xi1 := max(a.X1, b.X1)
	yi1 := max(a.Y1, b.Y1)
	xi2 := min(a.X2, b.X2)
	yi2 := min(a.Y2, b.Y2)
	intersect := max(float32(0), xi2-xi1) * max(float32(0), yi2-yi1)
	areaA := (a.X2 - a.X1) * (a.Y2 - a.Y1)
	areaB := (b.X2 - b.X1) * (b.Y2 - b.Y1)
	union := areaA + areaB - intersect
	if union == 0 {
		return 0
	}
	return intersect / union
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ---------- 保存为 YOLO txt 格式 ----------
func saveYOLOTxt(boxes []BBox, origW, origH int, txtPath string) error {
	f, err := os.Create(txtPath)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, b := range boxes {
		cx := (b.X1 + b.X2) / 2 / float32(origW)
		cy := (b.Y1 + b.Y2) / 2 / float32(origH)
		w := (b.X2 - b.X1) / float32(origW)
		h := (b.Y2 - b.Y1) / float32(origH)
		fmt.Fprintf(f, "%d %.6f %.6f %.6f %.6f %.6f\n", b.ClassID, cx, cy, w, h, b.Confidence)
	}
	return nil
}

// ---------- 绘制标注图片 ----------
func saveAnnotatedImg(img image.Image, boxes []BBox, outPath string) error {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	colors := map[int][3]uint8{
		0: {0, 0, 255}, 1: {255, 165, 0}, 2: {0, 255, 0}, 3: {255, 255, 0},
		5: {0, 255, 255}, 7: {255, 0, 0},
	}
	defaultColor := [3]uint8{128, 128, 128}

	for _, b := range boxes {
		c := defaultColor
		if v, ok := colors[b.ClassID]; ok {
			c = v
		}
		col := color.RGBA{c[0], c[1], c[2], 255}

		// 画框
		for y := int(b.Y1); y <= int(b.Y2); y++ {
			if y < 0 || y >= bounds.Dy() {
				continue
			}
			if int(b.X1) >= 0 && int(b.X1) < bounds.Dx() {
				rgba.Set(int(b.X1), y, col)
			}
			if int(b.X2) >= 0 && int(b.X2) < bounds.Dx() {
				rgba.Set(int(b.X2), y, col)
			}
		}
		for x := int(b.X1); x <= int(b.X2); x++ {
			if x < 0 || x >= bounds.Dx() {
				continue
			}
			if int(b.Y1) >= 0 && int(b.Y1) < bounds.Dy() {
				rgba.Set(x, int(b.Y1), col)
			}
			if int(b.Y2) >= 0 && int(b.Y2) < bounds.Dy() {
				rgba.Set(x, int(b.Y2), col)
			}
		}
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, rgba, &jpeg.Options{Quality: 90})
}

// ---------- 主函数 ----------
func main() {
	flag.Parse()

	fmt.Printf("Go 检测器 - 一致性对比\n")
	fmt.Printf("  模型: %s\n", *modelPath)
	fmt.Printf("  图片: %s\n", *imgPath)
	fmt.Printf("  参数: conf=%.2f iou=%.2f size=%d\n\n", *confThresh, *iouThresh, *inputSize)

	// 加载图片
	f, err := os.Open(*imgPath)
	if err != nil {
		fmt.Printf("错误: 无法打开图片 %s: %v\n", *imgPath, err)
		os.Exit(1)
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		fmt.Printf("错误: 解码图片失败: %v\n", err)
		os.Exit(1)
	}

	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()
	fmt.Printf("  图片尺寸: %dx%d\n\n", origW, origH)

	// 推理
	boxes, _, err := infer(img)
	if err != nil {
		fmt.Printf("错误: 推理失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("  检测到 %d 个目标:\n", len(boxes))
	chineseLabels := map[string]string{
		"person": "人员", "car": "汽车", "motorcycle": "摩托车", "bus": "巴士", "truck": "卡车",
	}
	for i, b := range boxes {
		ch := chineseLabels[b.Label]
		if ch == "" {
			ch = b.Label
		}
		fmt.Printf("    [%d] %s(%s) conf=%.4f box=[%.1f %.1f %.1f %.1f]\n",
			i+1, b.Label, ch, b.Confidence, b.X1, b.Y1, b.X2, b.Y2)
	}

	// 保存 YOLO 格式 txt
	if err := saveYOLOTxt(boxes, origW, origH, *outTxt); err != nil {
		fmt.Printf("错误: 保存 txt 失败: %v\n", err)
	} else {
		abs, _ := filepath.Abs(*outTxt)
		fmt.Printf("\n  检测结果已保存至: %s\n", abs)
	}

	// 保存标注图片
	if err := saveAnnotatedImg(img, boxes, *outImg); err != nil {
		fmt.Printf("错误: 保存图片失败: %v\n", err)
	} else {
		abs, _ := filepath.Abs(*outImg)
		fmt.Printf("  标注图片已保存至: %s\n", abs)
	}
}

var cocoNames = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
