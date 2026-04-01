package main

import (
	"fmt"
	"time"

	"yolo-go-detector/engine"
)

func main() {
	fmt.Println("=== 批量推理引擎示例 ===")

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

	// 模拟批量推理任务
	const taskCount = 10
	
	// 提交任务
	for i := 0; i < taskCount; i++ {
		// 模拟输入数据（实际应用中需要从图像中提取）
		inputData := make([]float32, 3*inputSize*inputSize)
		
		callback := make(chan *engine.InferenceResult, 1)
		
		task := &engine.InferenceTask{
			ImageData: inputData,
			Callback:  callback,
			Timeout:   timeout,
		}

		err := batchEngine.SubmitTask(task)
		if err != nil {
			fmt.Printf("提交任务 %d 失败：%v\n", i, err)
			continue
		}

		// 等待结果
		select {
		case result := <-callback:
			if result.Error != nil {
				fmt.Printf("任务 %d 执行失败：%v\n", i, result.Error)
			} else {
				fmt.Printf("任务 %d 执行成功\n", i)
			}
		case <-time.After(timeout):
			fmt.Printf("任务 %d 超时\n", i)
		}
	}

	fmt.Printf("\n批量推理完成 - 总任务数：%d\n", taskCount)
	fmt.Println("=== 示例完成 ===")
}
