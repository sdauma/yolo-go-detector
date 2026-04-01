package detector

import (
	"fmt"
	"time"

	"yolo-go-detector/engine"
)

type Detector interface {
	Detect(input []byte) (*engine.DetectionResult, error)
	DetectBatch(inputs [][]byte) ([]engine.DetectionResult, error)
	Close() error
	GetModelName() string
	GetInputShape() []int64
	GetOutputShape() []int64
}

type YOLO11xDetector struct {
	engine        *engine.BatchInferenceEngine
	postprocessor *engine.Postprocessor
	modelName     string
}

func NewYOLO11xDetector(modelPath string, confidenceThreshold, iouThreshold float64, inputSize int, useRectScaling, useAugment bool) *YOLO11xDetector {
	// 使用新的 BatchInferenceEngine API
	batchEngine := engine.NewBatchInferenceEngine(
		4,    // workerCount
		4,    // maxSessions
		modelPath,
		inputSize,
		1,    // batchSize
		30*time.Second, // timeout
	)

	nmsConfig := engine.DefaultNMSConfig()
	postprocessor := engine.NewPostprocessor(nmsConfig)

	return &YOLO11xDetector{
		engine:        batchEngine,
		postprocessor: postprocessor,
		modelName:     "YOLO11x",
	}
}

func (d *YOLO11xDetector) Detect(input []byte) (*engine.DetectionResult, error) {
	// TODO: 实现从图像字节到检测结果的转换
	// 这里需要实现图像预处理、推理、后处理的完整流程
	return nil, fmt.Errorf("未实现")
}

func (d *YOLO11xDetector) DetectBatch(inputs [][]byte) ([]engine.DetectionResult, error) {
	var results []engine.DetectionResult

	for _, input := range inputs {
		result, err := d.Detect(input)
		if err != nil {
			return nil, err
		}
		results = append(results, *result)
	}

	return results, nil
}

func (d *YOLO11xDetector) DetectWithPostprocess(input []float32) ([]engine.BoundingBox, error) {
	// TODO: 实现使用新的 BatchInferenceEngine API
	return nil, fmt.Errorf("未实现")
}

func (d *YOLO11xDetector) DetectBatchWithPostprocess(inputs [][]float32) ([][]engine.BoundingBox, error) {
	// TODO: 实现使用新的 BatchInferenceEngine API
	return nil, fmt.Errorf("未实现")
}

func (d *YOLO11xDetector) Close() error {
	d.engine.Stop()
	return nil
}

func (d *YOLO11xDetector) GetModelName() string {
	return d.modelName
}

func (d *YOLO11xDetector) GetInputShape() []int64 {
	return []int64{1, 3, 640, 640}
}

func (d *YOLO11xDetector) GetOutputShape() []int64 {
	return []int64{1, 84, 8400}
}

func (d *YOLO11xDetector) GetEngine() *engine.BatchInferenceEngine {
	return d.engine
}

func (d *YOLO11xDetector) GetPostprocessor() *engine.Postprocessor {
	return d.postprocessor
}
