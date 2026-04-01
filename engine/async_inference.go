package engine

import (
	"fmt"
	"sync"
	"time"
)

type AsyncInferencePipeline struct {
	inputChan  chan []float32
	resultChan chan DetectionResult
	pool       *SessionPool
	workers    int
	wg         sync.WaitGroup
}

func NewAsyncInferencePipeline(modelPath string, workers int, bufferSize int) (*AsyncInferencePipeline, error) {
	// 使用新的 SessionPool API
	pool := NewSessionPool(workers, modelPath, 640, 1)

	pipeline := &AsyncInferencePipeline{
		inputChan:  make(chan []float32, bufferSize),
		resultChan: make(chan DetectionResult, bufferSize),
		pool:       pool,
		workers:    workers,
	}

	pipeline.startWorkers()

	return pipeline, nil
}

func (p *AsyncInferencePipeline) startWorkers() {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go func(workerID int) {
			defer p.wg.Done()

			session, err := p.pool.GetSession()
			if err != nil {
				fmt.Printf("Worker %d: Failed to get session: %v\n", workerID, err)
				return
			}
			defer p.pool.PutSession(session)

			for input := range p.inputChan {
				copy(session.Input.GetData(), input)

				startTime := time.Now()
				err := session.Session.Run()
				latency := float64(time.Since(startTime).Microseconds()) / 1000.0

				result := DetectionResult{
					WorkerID:  workerID,
					Latency:   latency,
					Timestamp: time.Now().UnixMicro(),
				}

				if err == nil {
					outputData := make([]float32, len(session.Output.GetData()))
					copy(outputData, session.Output.GetData())
					result.Output = outputData
					result.Success = true
				} else {
					result.Error = fmt.Errorf("推理失败：%v", err)
					result.Success = false
				}

				p.resultChan <- result
			}
		}(i)
	}
}

func (p *AsyncInferencePipeline) Submit(input []float32) {
	p.inputChan <- input
}

func (p *AsyncInferencePipeline) Results() <-chan DetectionResult {
	return p.resultChan
}

func (p *AsyncInferencePipeline) SubmitBatch(inputs [][]float32) {
	for _, input := range inputs {
		p.Submit(input)
	}
}

func (p *AsyncInferencePipeline) Close() {
	close(p.inputChan)
	p.wg.Wait()
	close(p.resultChan)
	// SessionPool 不需要显式关闭，会话会在池中复用
}

func (p *AsyncInferencePipeline) Stats() (inputBuffered, resultBuffered int) {
	return len(p.inputChan), len(p.resultChan)
}
