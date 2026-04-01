package main

import (
	"fmt"
	"time"

	"yolo-go-detector/engine"
)

func main() {
	fmt.Println("=== 实时检测器示例 ===")

	// 配置参数
	modelPath := "./third_party/yolo11x.onnx"
	workerCount := 4
	maxSessions := 4
	inputSize := 640
	batchSize := 1
	timeout := 30 * time.Second

	// 创建批量推理引擎
	batchEngine := engine.NewBatchInferenceEngine(workerCount, maxSessions, modelPath, inputSize, batchSize, timeout)
	defer batchEngine.Stop()

	fmt.Println("批量推理引擎已创建")
	fmt.Printf("工作协程数：%d, 最大会话数：%d\n", workerCount, maxSessions)

	// 模拟实时检测任务
	fmt.Println("\n=== 实时检测模拟 ===")
	for i := 0; i < 5; i++ {
		startTime := time.Now()

		// 创建推理任务
		inputData := make([]float32, 3*inputSize*inputSize)
		callback := make(chan *engine.InferenceResult, 1)

		task := &engine.InferenceTask{
			ImageData: inputData,
			Callback:  callback,
			Timeout:   timeout,
		}

		err := batchEngine.SubmitTask(task)
		if err != nil {
			fmt.Printf("第 %d 次检测 - 提交任务失败：%v\n", i+1, err)
			continue
		}

		// 等待结果
		select {
		case result := <-callback:
			elapsed := time.Since(startTime)
			if result.Error != nil {
				fmt.Printf("第 %d 次检测 - 耗时：%v, 检测失败：%v\n", i+1, elapsed, result.Error)
			} else {
				fmt.Printf("第 %d 次检测 - 耗时：%v, 检测成功\n", i+1, elapsed)
			}
		case <-time.After(timeout):
			fmt.Printf("第 %d 次检测 - 超时\n", i+1)
		}

		time.Sleep(500 * time.Millisecond) // 模拟实时场景中的帧间隔
	}

	fmt.Println("\n=== 示例完成 ===")
}
