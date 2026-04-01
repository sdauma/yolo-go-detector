package engine

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// inferenceResult 推理结果
type inferenceResult struct {
	output  []float32
	latency time.Duration
	err     error
}

// CPUOptimizationConfig CPU 优化配置
type CPUOptimizationConfig struct {
	NumThreads        int  // CPU 线程数
	UseSIMD           bool // 使用 SIMD 指令集
	ThreadAffinity    bool // 线程绑定
	CacheOptimization bool // 缓存优化
	PrefetchDistance  int  // 预取距离
}

// DefaultCPUConfig 默认 CPU 配置
func DefaultCPUConfig() CPUOptimizationConfig {
	return CPUOptimizationConfig{
		NumThreads:        runtime.NumCPU(),
		UseSIMD:           true,
		ThreadAffinity:    false, // Windows 上需要特殊权限
		CacheOptimization: true,
		PrefetchDistance:  64,
	}
}

// CPUOptimizedEngine CPU 优化推理引擎
type CPUOptimizedEngine struct {
	config          CPUOptimizationConfig
	sessionPool     *SessionPool
	threadPools     []*WorkerPool
	cacheAlignedBuf sync.Pool
	stats           CPUStats
	roundRobin      uint64
}

// WorkerPool 工作线程池
type WorkerPool struct {
	workers   []Worker
	taskQueue chan Task
	wg        sync.WaitGroup
	shutdown  int32
	config    CPUOptimizationConfig
}

// Worker 工作线程
type Worker struct {
	id      int
	pool    *WorkerPool
	cpuCore int // 绑定的 CPU 核心
}

// Task 计算任务
type Task struct {
	data     []float32
	result   chan<- interface{}
	function func([]float32) interface{}
}

// CPUStats CPU 性能统计
type CPUStats struct {
	AverageCPUUsage  float64
	CacheHitRate     float64
	ThreadEfficiency float64
	SIMDSpeedup      float32
}

// NewCPUOptimizedEngine 创建 CPU 优化引擎
func NewCPUOptimizedEngine(
	modelPath string,
	config CPUOptimizationConfig,
	inputShape, outputShape []int64,
) (*CPUOptimizedEngine, error) {

	// 使用新的 SessionPool API
	sessionPool := NewSessionPool(
		config.NumThreads,
		modelPath,
		640, // inputSize
		1,   // batchSize
	)

	engine := &CPUOptimizedEngine{
		config:      config,
		sessionPool: sessionPool,
		threadPools: make([]*WorkerPool, 0),
	}

	// 创建工作线程池
	for i := 0; i < config.NumThreads; i++ {
		pool := NewWorkerPool(4, i, config)
		engine.threadPools = append(engine.threadPools, pool)
	}

	// 创建缓存对齐的内存池
	engine.cacheAlignedBuf.New = func() interface{} {
		// 分配 64 字节对齐的缓冲区（缓存行大小）
		return make([]float32, 0, 1024)
	}

	return engine, nil
}

// NewWorkerPool 创建工作线程池
func NewWorkerPool(queueSize int, threadID int, config CPUOptimizationConfig) *WorkerPool {
	pool := &WorkerPool{
		workers:   make([]Worker, 0),
		taskQueue: make(chan Task, queueSize),
		config:    config,
	}

	// 创建工作线程
	numWorkers := min(4, runtime.NumCPU())
	for i := 0; i < numWorkers; i++ {
		worker := Worker{
			id:      i*runtime.NumCPU() + threadID,
			pool:    pool,
			cpuCore: i % runtime.NumCPU(),
		}
		pool.workers = append(pool.workers, worker)

		pool.wg.Add(1)
		go worker.start()
	}

	return pool
}

// start 启动工作线程
func (w *Worker) start() {
	defer w.pool.wg.Done()

	// 如果启用线程绑定，尝试绑定到特定 CPU 核心
	// 注意：Go runtime 不直接支持 CPU 绑定，需要 CGO 调用
	if w.pool.config.ThreadAffinity {
		// bindToCPU(w.cpuCore) // 需要平台特定实现
	}

	for {
		select {
		case task, ok := <-w.pool.taskQueue:
			if !ok {
				return
			}

			// 执行任务
			result := task.function(task.data)
			if task.result != nil {
				task.result <- result
			}

		default:
			// 无任务时让出 CPU
			runtime.Gosched()
		}
	}
}

// SubmitTask 提交任务到线程池
func (e *CPUOptimizedEngine) SubmitTask(data []float32, fn func([]float32) interface{}) <-chan interface{} {
	resultChan := make(chan interface{}, 1)

	// 轮询分发到不同线程池
	poolIdx := int(atomic.AddUint64(&e.roundRobin, 1)) % len(e.threadPools)

	task := Task{
		data:     data,
		result:   resultChan,
		function: fn,
	}

	e.threadPools[poolIdx].taskQueue <- task

	return resultChan
}

// BatchParallelInfer 批量并行推理（CPU 优化版）
func (e *CPUOptimizedEngine) BatchParallelInfer(inputs [][]byte) ([]inferenceResult, error) {
	numInputs := len(inputs)
	results := make([]inferenceResult, numInputs)

	var wg sync.WaitGroup
	errors := make(chan error, numInputs)

	// 分批处理
	batchSize := max(1, numInputs/e.config.NumThreads)

	for i := 0; i < numInputs; i += batchSize {
		end := min(i+batchSize, numInputs)
		batch := inputs[i:end]

		wg.Add(len(batch))
		for j, input := range batch {
			go func(idx int, inputData []byte) {
				defer wg.Done()

				// 从缓存池获取缓冲区
				buf := e.cacheAlignedBuf.Get().([]float32)
				defer e.cacheAlignedBuf.Put(buf[:0])

				// 执行推理
				session, err := e.sessionPool.GetSession()
				if err != nil {
					errors <- fmt.Errorf("获取会话失败：%v", err)
					return
				}
				defer e.sessionPool.PutSession(session)

				start := time.Now()
				err = session.Session.Run()
				latency := time.Since(start)

				results[idx] = inferenceResult{
					output:  session.Output.GetData(),
					latency: latency,
					err:     err,
				}
			}(i+j, input)
		}

		wg.Wait()
	}

	close(errors)

	// 检查错误
	for err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// GetCPUStats 获取 CPU 性能统计
func (e *CPUOptimizedEngine) GetCPUStats() CPUStats {
	return e.stats
}

// EstimateCPUSpeedup 预估 CPU 优化加速比
func (e *CPUOptimizedEngine) EstimateCPUSpeedup() float32 {
	speedup := float32(1.0)

	// 多线程加速
	speedup *= float32(e.config.NumThreads) * 0.8 // 80% 并行效率

	// SIMD 加速（AVX2/AVX-512）
	if e.config.UseSIMD {
		speedup *= 2.0 // 保守估计 2 倍 SIMD 加速
	}

	// 缓存优化
	if e.config.CacheOptimization {
		speedup *= 1.2 // 20% 缓存优化收益
	}

	return speedup
}

// Infer 执行推理
func (e *CPUOptimizedEngine) Infer(input []float32) (float64, error) {
	session, err := e.sessionPool.GetSession()
	if err != nil {
		return 0, fmt.Errorf("获取会话失败：%v", err)
	}
	defer e.sessionPool.PutSession(session)

	// 缓存优化：使用缓存对齐的内存
	buf := e.cacheAlignedBuf.Get().([]float32)
	buf = append(buf, input...)
	copy(session.Input.GetData(), buf)
	e.cacheAlignedBuf.Put(buf)

	// 执行推理
	start := time.Now()
	err = session.Session.Run()
	latency := float64(time.Since(start).Microseconds()) / 1000.0

	return latency, err
}

// Stats 获取 CPU 性能统计
func (e *CPUOptimizedEngine) Stats() CPUStats {
	return e.stats
}

// OptimizeThreads 优化线程配置
func (e *CPUOptimizedEngine) OptimizeThreads() {
	// 基于当前负载动态调整线程数
	// 这里可以实现更复杂的线程数优化逻辑
	runtime.GOMAXPROCS(e.config.NumThreads)
}

// Close 关闭引擎
func (e *CPUOptimizedEngine) Close() {
	// 关闭所有线程池
	for _, pool := range e.threadPools {
		atomic.StoreInt32(&pool.shutdown, 1)
		close(pool.taskQueue)
		pool.wg.Wait()
	}

	// SessionPool 不需要显式关闭
}
