package main

import (
	"fmt"
	"sync"
	"time"

	"yolo-go-detector/engine"
)

func main() {
	fmt.Println("=== 基本 Session Pool 示例 ===")

	// 配置参数
	modelPath := "./third_party/yolo11x.onnx"
	maxSessions := 4
	inputSize := 640
	batchSize := 1

	// 创建会话池
	pool := engine.NewSessionPool(maxSessions, modelPath, inputSize, batchSize)
	// 注意：SessionPool 会在内部管理会话生命周期

	// 测试会话获取和归还 - 控制并发量不超过 maxSessions
	const workerCount = 4
	var wg sync.WaitGroup
		for i := 0; i < workerCount; i++ {
			wg.Add(1)
			go func(index int) {
			// 获取会话
			session, err := pool.GetSession()
			if err != nil {
				fmt.Printf("Worker %d: 获取会话失败: %v\n", index, err)
				return
			}
			defer pool.PutSession(session)

			// 模拟推理操作
			time.Sleep(100 * time.Millisecond)
			fmt.Printf("Worker %d: 完成推理\n", index)

			// 获取会话池统计信息
			active, idle := pool.GetStats()
			fmt.Printf("Worker %d: 会话池状态 - 活跃: %d, 空闲: %d\n", index, active, idle)
		}(i)
	}

	// 等待所有协程完成
	wg.Wait()

	// 最终统计信息
	active, idle := pool.GetStats()
	fmt.Printf("\n最终会话池状态 - 活跃: %d, 空闲: %d\n", active, idle)
	fmt.Println("=== 示例完成 ===")
}
