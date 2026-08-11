// Package main 演示 Go 与 Python ONNX Runtime 推理结果的一致性验证。
//
// 论文 §3.3 声明：Go 侧推理结果与 Python 官方 ONNX Runtime 推理结果高度一致。
// 本示例展示验证方法——导出原始输出张量，供与 Python 侧对比。
//
// 运行本示例后，再运行 test/python/python_output_consistency.py，
// 即可对比两侧原始输出张量的逐元素差异（预期 max|diff| < 1e-4）。
package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"math"
	"os"
	"path/filepath"

	"yolo-go-detector/engine"

	"github.com/nfnt/resize"
)

// repoRoot 从当前工作目录向上查找 go.mod，定位仓库根目录，
// 使本示例无论以何种方式运行（go run ./examples/consistency 或 cd 后 go run .）
// 都能正确定位 third_party / assets / results。
func repoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "."
}

func main() {
	fmt.Println("=== Go-Python 推理一致性验证示例 ===")
	fmt.Println()

	root := repoRoot()
	engine.SetONNXLibPath(filepath.Join(root, "third_party", "onnxruntime.dll"))
	outputDir := filepath.Join(root, "results")
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("创建输出目录失败: %v\n", err)
		os.Exit(1)
	}

	models := []struct {
		name string
		file string
	}{
		{"yolo11x", "yolo11x.onnx"},
		{"yolo11n", "yolo11n.onnx"},
	}

	for _, m := range models {
		modelPath := filepath.Join(root, "third_party", m.file)
		imagePath := filepath.Join(root, "assets", "bus.jpg")
		if err := runModel(modelPath, imagePath, outputDir, m.name); err != nil {
			fmt.Printf("模型 %s 验证失败: %v\n", m.name, err)
		}
	}

	fmt.Println()
	fmt.Println("=== 如何验证 Go-Python 一致性 ===")
	fmt.Println("1. Go 侧（本程序）已导出原始输出张量到 results/go_<model>_output.bin")
	fmt.Println("   格式: 连续 float32 小端序，形状 [1, 84, 8400]")
	fmt.Println("2. 运行 Python 侧:  python test/python/python_output_consistency.py")
	fmt.Println("   Python 会在 results/ 下生成 python_yolo11x_detections.txt / python_yolo11n_detections.txt")
	fmt.Println("3. 在 Python 中加载两侧原始张量比对:")
	fmt.Println("   import numpy as np")
	fmt.Println("   go_output = np.fromfile('results/go_yolo11x_output.bin', dtype=np.float32)")
	fmt.Println("   go_output = go_output.reshape(1, 84, 8400)")
	fmt.Println("   # 与 python_output_consistency.py 的 output[0] 对比")
	fmt.Println("   print('max|diff| =', np.max(np.abs(go_output - py_output)))")
	fmt.Println("   # 预期 max|diff| < 1e-4（浮点精度等价）")
	fmt.Println()
	fmt.Println("=== 示例完成 ===")
}

func runModel(modelPath, imagePath, outputDir, modelName string) error {
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("模型文件不存在: %s", modelPath)
	}
	if _, err := os.Stat(imagePath); err != nil {
		return fmt.Errorf("测试图片不存在: %s", imagePath)
	}

	img, origW, origH := loadImage(imagePath)
	tensor, scaleInfo := preprocessLetterbox(img, 640)

	fmt.Printf("\n--- 模型 %s ---\n", modelName)
	fmt.Printf("原始图像: %dx%d\n", origW, origH)
	fmt.Printf("预处理: 640x640 Letterbox (scale=%.4f, padLeft=%.0f, padTop=%.0f)\n",
		scaleInfo.scale, scaleInfo.padLeft, scaleInfo.padTop)

	// 论文 §5.2 主推荐配置：单 Session 顺序推理 + 显式 6 线程
	pool := engine.NewSessionPool(1, modelPath, 640, 1, 6)
	session, err := pool.GetSession()
	if err != nil {
		return fmt.Errorf("获取 Session 失败: %w", err)
	}
	defer pool.PutSession(session)

	exportInputTensor(tensor, outputDir, modelName)
	copy(session.Input.GetData(), tensor)
	if err := session.Session.Run(); err != nil {
		return fmt.Errorf("推理失败: %w", err)
	}

	outputData := session.Output.GetData()
	exportRawOutput(outputData, outputDir, modelName)

	pp := engine.NewPostprocessor(engine.NMSConfig{
		ConfThreshold: 0.25,
		IoUThreshold:  0.7,
		MaxDetections: 300,
	})
	boxes := pp.Process(outputData, 640, 640)

	scaledBoxes := pp.ScaleBboxes(boxes, scaleInfo.scale, scaleInfo.scale,
		scaleInfo.padLeft, scaleInfo.padTop)

	fmt.Printf("检测到 %d 个对象:\n", len(scaledBoxes))
	for i, box := range scaledBoxes {
		fmt.Printf("  目标 %d: 类别=%d(%s), 置信度=%.4f, 坐标=(%.3f, %.3f, %.3f, %.3f)\n",
			i+1, box.ClassID, engine.GetClassName(box.ClassID), box.Confidence,
			box.XMin, box.YMin, box.XMax, box.YMax)
	}

	return nil
}

func loadImage(path string) (image.Image, int, int) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Printf("打开图片失败: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		fmt.Printf("解码图片失败: %v\n", err)
		os.Exit(1)
	}

	bounds := img.Bounds()
	return img, bounds.Dx(), bounds.Dy()
}

type scaleInfo struct {
	scale   float32
	padLeft float32
	padTop  float32
}

// preprocessLetterbox 与 Python 侧 resize_with_letterbox 完全一致的预处理：
// 等比缩放到 640、灰色 (114) 填充、RGB 归一化到 [0,1]、布局 [3,640,640]。
func preprocessLetterbox(img image.Image, targetSize int) ([]float32, scaleInfo) {
	bounds := img.Bounds()
	ow, oh := bounds.Dx(), bounds.Dy()

	scale := math.Min(float64(targetSize)/float64(ow), float64(targetSize)/float64(oh))
	nw := int(math.Round(float64(ow) * scale))
	nh := int(math.Round(float64(oh) * scale))

	resized := resize.Resize(uint(nw), uint(nh), img, resize.Bilinear)

	result := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	draw.Draw(result, result.Bounds(),
		&image.Uniform{color.RGBA{114, 114, 114, 255}},
		image.Point{}, draw.Src)

	offsetX := (targetSize - nw) / 2
	offsetY := (targetSize - nh) / 2
	draw.Draw(result, image.Rect(offsetX, offsetY, offsetX+nw, offsetY+nh),
		resized, image.Point{}, draw.Src)

	channelSize := targetSize * targetSize
	tensor := make([]float32, 3*channelSize)
	r := tensor[:channelSize]
	g := tensor[channelSize : 2*channelSize]
	b := tensor[2*channelSize : 3*channelSize]

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			idx := y*targetSize + x
			pr, pg, pb, _ := result.At(x, y).RGBA()
			r[idx] = float32(pr>>8) / 255.0
			g[idx] = float32(pg>>8) / 255.0
			b[idx] = float32(pb>>8) / 255.0
		}
	}

	return tensor, scaleInfo{
		scale:   float32(scale),
		padLeft: float32(offsetX),
		padTop:  float32(offsetY),
	}
}

func exportRawOutput(data []float32, dir, modelName string) {
	path := filepath.Join(dir, fmt.Sprintf("go_%s_output.bin", modelName))
	f, err := os.Create(path)
	if err != nil {
		fmt.Printf("创建输出文件失败: %v\n", err)
		return
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, data); err != nil {
		fmt.Printf("写入输出文件失败: %v\n", err)
		return
	}

	fmt.Printf("原始输出张量已导出: %s (%d 个 float32, 形状 [1, 84, 8400])\n", path, len(data))
}

// exportInputTensor 导出预处理后的输入张量（与 Python 侧 resize_with_letterbox 一致的 [3,640,640] NCHW，
// float32 小端序），供 python_output_consistency.py 的 compare_with_go_export 加载并复用同一输入张量推理，
// 比对两侧 max|diff|（预期 < 1e-4）。这是方案B的核心步骤——Go 与 Python 喂入完全相同的张量后再比输出。
func exportInputTensor(data []float32, dir, modelName string) {
	path := filepath.Join(dir, fmt.Sprintf("go_%s_input.bin", modelName))
	f, err := os.Create(path)
	if err != nil {
		fmt.Printf("创建输入文件失败: %v\n", err)
		return
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, data); err != nil {
		fmt.Printf("写入输入文件失败: %v\n", err)
		return
	}

	fmt.Printf("原始输入张量已导出: %s (%d 个 float32, 形状 [1, 3, 640, 640] NCHW)\n", path, len(data))
}
