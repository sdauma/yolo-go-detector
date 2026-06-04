package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"yolo-go-detector/engine"

	"github.com/nfnt/resize"
	"golang.org/x/image/font"
	"golang.org/x/image/font/inconsolata"
	"golang.org/x/image/math/fixed"
)

// DetectionPipeline YOLO 检测流水线
// 整合预处理、推理、后处理全流程
type DetectionPipeline struct {
	cfg  *Config
	pool *engine.SessionPool
	post *engine.Postprocessor
}

// NewDetectionPipeline 创建检测流水线
func NewDetectionPipeline(cfg *Config) (*DetectionPipeline, error) {
	poolSize := cfg.Detection.PoolSize
	if poolSize <= 0 {
		numCPU := runtime.NumCPU()
		// 大模型（yolo11x）每 Session 至少需要 3-6 线程才能有效运行。
		// 策略：池大小 = max(1, CPU/4)，确保 intraOp ≥ 4；上限 3（避免单 Session 线程太少）。
		// 用户可通过 pool_size 手动覆盖（如设置为 2 在大模型上通常最优）。
		poolSize = max(1, numCPU/4)
		if poolSize > 3 {
			poolSize = 3
		}
	}
	intraOp := cfg.Detection.IntraOpThreads
	if intraOp <= 0 {
		intraOp = max(1, runtime.NumCPU()/poolSize)
	}

	// 大模型告警：每 Session 不足 3 线程时推理显著变慢
	if intraOp < 3 && poolSize > 1 {
		numCPU := runtime.NumCPU()
		fmt.Printf("[提示] 每 Session 仅 %d 线程，大模型(yolo11x)可能较慢。可尝试 pool_size=%d\n",
			intraOp, max(1, numCPU/6))
	}

	pool := engine.NewSessionPool(
		poolSize,
		cfg.Detection.ModelPath,
		cfg.Detection.InputSize,
		1, // batchSize=1，每路单独推理
		intraOp, // 自动计算 = CPU/PoolSize，确保合理分配
	)

	return &DetectionPipeline{
		cfg:  cfg,
		pool: pool,
		post: engine.NewPostprocessor(engine.NMSConfig{
			ConfThreshold: float32(cfg.Detection.ConfThresh),
			IoUThreshold:  float32(cfg.Detection.IOUThresh),
			MaxDetections: 300,
		}),
	}, nil
}

// DetectResult 单次检测结果
type DetectResult struct {
	ChannelCode string          `json:"channel_code"`
	ChannelName string          `json:"channel_name"`
	OrgName     string          `json:"org_name"`
	Timestamp   string          `json:"timestamp"`
	Width       int             `json:"width"`
	Height      int             `json:"height"`
	Detections  []DetectionItem `json:"detections"`
	Error       string          `json:"error,omitempty"`
	FetchMs     int64           `json:"fetch_ms"`
	InferMs     int64           `json:"infer_ms"`
}

// DetectionItem 单个检测对象
type DetectionItem struct {
	Class      string  `json:"class"`
	Confidence float32 `json:"confidence"`
	XMin       float32 `json:"xmin"`
	YMin       float32 `json:"ymin"`
	XMax       float32 `json:"xmax"`
	YMax       float32 `json:"ymax"`
}

// Detect 对单张图片执行检测（图片字节流 → 检测结果，全程不落盘）
func (dp *DetectionPipeline) Detect(cam *CameraInfo, imgData []byte) *DetectResult {
	result := &DetectResult{
		ChannelCode: cam.ChannelCode,
		ChannelName: cam.ChannelName,
		OrgName:     cam.OrgName,
		Timestamp:   time.Now().Format("2006-01-02 15:04:05"),
	}

	t0 := time.Now()

	// 1. 解码图片
	img, err := jpeg.Decode(bytes.NewReader(imgData))
	if err != nil {
		// 尝试 PNG
		img2, err2 := decodeImageAny(bytes.NewReader(imgData))
		if err2 != nil {
			result.Error = fmt.Sprintf("decode: %v", err)
			return result
		}
		img = img2
	}

	result.Width = img.Bounds().Dx()
	result.Height = img.Bounds().Dy()
	result.FetchMs = time.Since(t0).Milliseconds()

	// 2. 预处理：缩放 + 归一化 → float32 tensor，同时获取得 ScaleInfo
	t1 := time.Now()
	tensorData, scaleInfo := preprocessImage(img, dp.cfg.Detection.InputSize)

	// 3. 从 Session Pool 获取会话并推理
	session, err := dp.pool.GetSession()
	if err != nil {
		result.Error = fmt.Sprintf("get_session: %v", err)
		return result
	}

	copy(session.Input.GetData(), tensorData)

	runErr := session.Session.Run()
	if runErr != nil {
		dp.pool.PutSession(session)
		result.Error = fmt.Sprintf("infer: %v", runErr)
		return result
	}

	// 4. 后处理：NMS（框坐标在 640x640 输入空间）
	outputData := session.Output.GetData()
	dp.pool.PutSession(session)

	boxes := dp.post.Process(outputData, dp.cfg.Detection.InputSize, dp.cfg.Detection.InputSize)

	// 5. 映射回原图坐标（与 root/main.go 已验证的算法一致）
	// 公式: origCoord = (coord - Pad) / Scale
	for _, box := range boxes {
		className := engine.GetClassName(box.ClassID)
		x1 := (box.XMin - scaleInfo.PadLeft) / scaleInfo.ScaleX
		y1 := (box.YMin - scaleInfo.PadTop) / scaleInfo.ScaleY
		x2 := (box.XMax - scaleInfo.PadLeft) / scaleInfo.ScaleX
		y2 := (box.YMax - scaleInfo.PadTop) / scaleInfo.ScaleY

		// clamp 到原图范围
		x1 = clampF32(x1, 0, float32(result.Width))
		y1 = clampF32(y1, 0, float32(result.Height))
		x2 = clampF32(x2, 0, float32(result.Width))
		y2 = clampF32(y2, 0, float32(result.Height))

		result.Detections = append(result.Detections, DetectionItem{
			Class:      className,
			Confidence: box.Confidence,
			XMin:       x1,
			YMin:       y1,
			XMax:       x2,
			YMax:       y2,
		})
	}

	result.InferMs = time.Since(t1).Milliseconds()
	return result
}

// DrawAlertImage 在图片上绘制检测框和标签并保存
// 仅当检测到 alertClasses 中的目标时才调用
// 使用 engine 包中的完整 80 类颜色映射和中文标签
func (dp *DetectionPipeline) DrawAlertImage(result *DetectResult, imgData []byte, outputPath string) error {
	img, _, err := image.Decode(bytes.NewReader(imgData))
	if err != nil {
		return fmt.Errorf("解码图片失败: %w", err)
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, image.Point{}, draw.Src)

	for _, det := range result.Detections {
		boxColor := engine.GetClassColor(det.Class)

		x1, y1 := int(det.XMin), int(det.YMin)
		x2, y2 := int(det.XMax), int(det.YMax)

		// 绘制边界框（2px 宽，与 root/main.go 风格一致）
		drawRect(rgba, x1, y1, x2, y2, boxColor)

		// 绘制标签（中文名 + 置信度）
		drawLabelOnImage(rgba, det.Class, det.Confidence, x1, y1, boxColor)
	}

	// 创建输出目录
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return jpeg.Encode(outFile, rgba, &jpeg.Options{Quality: 90})
}

// drawLabelOnImage 在图片上绘制标签文本（含中文标签和置信度）
// 自动根据背景颜色选择高对比度文本颜色
func drawLabelOnImage(rgba *image.RGBA, className string, confidence float32, x1, y1 int, boxColor color.RGBA) {
	chineseLabel := engine.GetChineseLabel(className)
	labelText := fmt.Sprintf("%s/%s(%.2f)", className, chineseLabel, confidence)

	face := getLabelFont()
	textWidth, textHeight := measureLabelText(labelText, face)
	bounds := rgba.Bounds()

	// 计算标签位置（在框的左上角上方）
	textX := x1 + 5
	textY := y1 - 5

	if textY < textHeight {
		textY = y1 + textHeight + 5
	}
	if textY > bounds.Dy()-5 {
		textY = y1 - textHeight - 5
		if textY < 5 {
			textY = y1 + 10
		}
	}
	if textX+textWidth > bounds.Dx()-5 {
		textX = bounds.Dx() - textWidth - 10
		if textX < 5 {
			textX = 5
		}
	}
	if textX < 5 {
		textX = 5
	}

	// 绘制背景矩形
	bgPadding := 6
	bgX := textX - bgPadding/2
	bgY := textY - textHeight + 2
	bgW := textWidth + bgPadding*2
	bgH := textHeight + 4

	// Clamp 背景到图像范围
	if bgX < 0 {
		bgX = 0
	}
	if bgY < 0 {
		bgY = 0
	}
	if bgX+bgW > bounds.Dx() {
		bgW = bounds.Dx() - bgX
	}
	if bgY+bgH > bounds.Dy() {
		bgH = bounds.Dy() - bgY
	}

	// 绘制半透明背景
	for dy := 0; dy < bgH; dy++ {
		for dx := 0; dx < bgW; dx++ {
			rgba.Set(bgX+dx, bgY+dy, boxColor)
		}
	}

	// 选择高对比度文本颜色
	textColor := getContrastTextColor(boxColor)

	// 绘制文本
	point := fixed.P(textX, textY)
	d := &font.Drawer{
		Dst:  rgba,
		Src:  image.NewUniform(textColor),
		Face: face,
		Dot:  point,
	}
	d.DrawString(labelText)
}

// getLabelFont 获取用于标签绘制的字体
func getLabelFont() font.Face {
	// 使用 inconsolata 作为默认等宽字体（无需系统字体依赖）
	return inconsolata.Regular8x16
}

// measureLabelText 测量文本在指定字体下的尺寸
func measureLabelText(text string, face font.Face) (width, height int) {
	if face == nil {
		// 中文字符宽 14，ASCII 字符宽 8
		w := 0
		for _, r := range text {
			if r > 127 {
				w += 14
			} else {
				w += 8
			}
		}
		return w, 16
	}
	drawer := &font.Drawer{Face: face}
	advance := drawer.MeasureString(text)
	width = advance.Round()
	metrics := face.Metrics()
	height = (metrics.Height + metrics.Descent).Round()
	return
}

// getContrastTextColor 根据背景颜色返回高对比度文本颜色
func getContrastTextColor(bgColor color.RGBA) color.RGBA {
	luminance := 0.299*float64(bgColor.R) + 0.587*float64(bgColor.G) + 0.114*float64(bgColor.B)
	if luminance > 128 {
		return color.RGBA{0, 0, 0, 255} // 深色文本
	}
	return color.RGBA{255, 255, 255, 255} // 浅色文本
}

// HasAlert 检查结果是否包含告警目标
func (dp *DetectionPipeline) HasAlert(result *DetectResult) bool {
	for _, det := range result.Detections {
		for _, alertClass := range dp.cfg.Output.AlertClasses {
			if det.Class == alertClass {
				return true
			}
		}
	}
	return false
}

// PoolStats 返回 Session Pool 统计
func (dp *DetectionPipeline) PoolStats() (active, idle int) {
	return dp.pool.GetStats()
}

// scaleInfo 图片缩放信息（与 root/main.go 算法一致，经 Python 官方验证）
type scaleInfo struct {
	ScaleX  float32
	ScaleY  float32
	PadLeft float32
	PadTop  float32
}

// preprocessImage 图片预处理：缩放 + 归一化
// 返回张量和缩放信息（用于正确的坐标映射）
func preprocessImage(img image.Image, inputSize int) ([]float32, scaleInfo) {
	// Letterbox 缩放
	resized, si := resizeWithLetterbox(img, inputSize)

	// 提取 RGB 通道并归一化到 [0, 1]
	channelSize := inputSize * inputSize
	tensor := make([]float32, 3*channelSize)
	r := tensor[:channelSize]
	g := tensor[channelSize : 2*channelSize]
	b := tensor[2*channelSize : 3*channelSize]

	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			idx := y*inputSize + x
			pr, pg, pb, _ := resized.At(x, y).RGBA()
			r[idx] = float32(pr>>8) / 255.0
			g[idx] = float32(pg>>8) / 255.0
			b[idx] = float32(pb>>8) / 255.0
		}
	}

	return tensor, si
}

// resizeWithLetterbox 标准 Letterbox 缩放（保持长宽比 + 灰色填充）
// 返回缩放后的图像和 ScaleInfo（与 root/main.go 算法完全一致）
func resizeWithLetterbox(img image.Image, targetSize int) (image.Image, scaleInfo) {
	bounds := img.Bounds()
	ow, oh := bounds.Dx(), bounds.Dy()

	// 官方逻辑：r = min(targetSize/ow, targetSize/oh)
	scale := math.Min(float64(targetSize)/float64(ow), float64(targetSize)/float64(oh))
	nw := int(math.Round(float64(ow) * scale))
	nh := int(math.Round(float64(oh) * scale))

	resized := resize.Resize(uint(nw), uint(nh), img, resize.Bilinear)

	result := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	// 灰色填充 (114, 114, 114) — YOLO 标准
	draw.Draw(result, result.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)

	offsetX := (targetSize - nw) / 2
	offsetY := (targetSize - nh) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+nw, offsetY+nh), resized, image.Point{}, draw.Src)

	return result, scaleInfo{
		ScaleX:  float32(scale),
		ScaleY:  float32(scale),
		PadLeft: float32(offsetX),
		PadTop:  float32(offsetY),
	}
}

// clampF32 限制 float32 值在 [min, max] 范围
func clampF32(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// drawRect 在 RGBA 图像上绘制矩形边框（2px 宽）
func drawRect(rgba *image.RGBA, x1, y1, x2, y2 int, c color.RGBA) {
	bounds := rgba.Bounds()
	clamp := func(v, lo, hi int) int {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
	x1, x2 = clamp(x1, 0, bounds.Dx()-1), clamp(x2, 0, bounds.Dx()-1)
	y1, y2 = clamp(y1, 0, bounds.Dy()-1), clamp(y2, 0, bounds.Dy()-1)

	// 上下边
	for x := x1; x <= x2; x++ {
		rgba.Set(x, y1, c)
		if y1+1 <= y2 {
			rgba.Set(x, y1+1, c)
		}
		rgba.Set(x, y2, c)
		if y2-1 >= y1 {
			rgba.Set(x, y2-1, c)
		}
	}
	// 左右边
	for y := y1; y <= y2; y++ {
		rgba.Set(x1, y, c)
		if x1+1 <= x2 {
			rgba.Set(x1+1, y, c)
		}
		rgba.Set(x2, y, c)
		if x2-1 >= x1 {
			rgba.Set(x2-1, y, c)
		}
	}
}

// decodeImageAny 尝试解码任意格式图片
func decodeImageAny(r io.Reader) (image.Image, error) {
	img, _, err := image.Decode(r)
	return img, err
}
