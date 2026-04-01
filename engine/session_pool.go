package engine

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// ModelSession 表示一个ONNX Runtime会话
type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

// Destroy 销毁会话和张量资源
func (m *ModelSession) Destroy() {
	if m.Input != nil {
		m.Input.Destroy()
	}
	if m.Output != nil {
		m.Output.Destroy()
	}
	if m.Session != nil {
		m.Session.Destroy()
	}
}

// ScaleInfo 缩放和填充信息结构体
type ScaleInfo struct {
	ScaleX    float32
	ScaleY    float32
	PadLeft   int
	PadTop    int
	NewWidth  int
	NewHeight int
}



// SessionPool 会话池
type SessionPool struct {
	sessions       chan *ModelSession
	maxSize        int
	activeSessions int32
	mutex          sync.Mutex
	modelPath      string
	inputSize      int
	batchSize      int
}

// NewSessionPool 创建新的会话池
func NewSessionPool(maxSize int, modelPath string, inputSize, batchSize int) *SessionPool {
	pool := &SessionPool{
		sessions:  make(chan *ModelSession, maxSize),
		maxSize:   maxSize,
		modelPath: modelPath,
		inputSize: inputSize,
		batchSize: batchSize,
	}

	// 预创建一些会话
	preCreateCount := max(1, min(maxSize/2, runtime.NumCPU()))
	for i := 0; i < preCreateCount; i++ {
		if session, err := pool.createSession(); err == nil {
			select {
			case pool.sessions <- session:
			default:
				session.Destroy()
			}
		}
	}

	return pool
}

// GetSession 从池中获取会话
func (pool *SessionPool) GetSession() (*ModelSession, error) {
	// 先尝试从空闲列表获取
	select {
	case session := <-pool.sessions:
		if session != nil && session.Session != nil {
			atomic.AddInt32(&pool.activeSessions, 1)
			return session, nil
		}
		if session != nil {
			session.Destroy()
		}
	default:
	}

	// 如果没有空闲会话，检查是否可以创建新会话
	currentActive := atomic.LoadInt32(&pool.activeSessions)
	if currentActive < int32(pool.maxSize) {
		// 使用 CAS 操作确保不会超过最大值
		if atomic.CompareAndSwapInt32(&pool.activeSessions, currentActive, currentActive+1) {
			return pool.createSession()
		}
		// 如果 CAS 失败，说明有其他 goroutine 抢先创建了会话，重试获取
		return pool.GetSession()
	}

	// 达到最大并发限制，返回错误
	return nil, fmt.Errorf("active sessions reached max capacity: %d", pool.maxSize)
}

// PutSession 将会话放回池中
func (pool *SessionPool) PutSession(session *ModelSession) {
	atomic.AddInt32(&pool.activeSessions, -1)

	if session == nil || session.Session == nil {
		return
	}

	select {
	case pool.sessions <- session:
	default:
		session.Destroy()
	}
}

// GetStats 获取会话池统计信息
func (pool *SessionPool) GetStats() (active, idle int) {
	active = int(atomic.LoadInt32(&pool.activeSessions))
	idle = len(pool.sessions)
	return
}

// createSession 创建新的会话（内部使用，不检查并发限制）
func (pool *SessionPool) createSession() (*ModelSession, error) {
	session, err := initSession(pool.modelPath, pool.inputSize, pool.batchSize)
	if err != nil {
		atomic.AddInt32(&pool.activeSessions, -1)
		return nil, err
	}

	return session, nil
}

// BatchInferenceEngine 批量推理引擎
type BatchInferenceEngine struct {
	sessionPool *SessionPool
	workerCount int
	taskQueue   chan *InferenceTask
	resultQueue chan *InferenceResult
	shutdown    chan struct{}
	wg          sync.WaitGroup
	timeout     time.Duration
}

// InferenceTask 推理任务
type InferenceTask struct {
	ImageData []float32
	Callback  chan<- *InferenceResult
	Timeout   time.Duration
}

// InferenceResult 推理结果
type InferenceResult struct {
	Boxes []BoundingBox
	Error error
}

// NewBatchInferenceEngine 创建新的批量推理引擎
func NewBatchInferenceEngine(workerCount int, maxSessions int, modelPath string, inputSize, batchSize int, timeout time.Duration) *BatchInferenceEngine {
	if workerCount > runtime.NumCPU()*2 {
		workerCount = runtime.NumCPU() * 2
	}

	if maxSessions > runtime.NumCPU()*2 {
		maxSessions = runtime.NumCPU() * 2
	}

	engine := &BatchInferenceEngine{
		sessionPool: NewSessionPool(maxSessions, modelPath, inputSize, batchSize),
		workerCount: workerCount,
		taskQueue:   make(chan *InferenceTask, 100),
		resultQueue: make(chan *InferenceResult, 100),
		shutdown:    make(chan struct{}),
		timeout:     timeout,
	}

	for i := 0; i < workerCount; i++ {
		engine.wg.Add(1)
		go engine.worker(i)
	}

	return engine
}

// SubmitTask 提交推理任务
func (engine *BatchInferenceEngine) SubmitTask(task *InferenceTask) error {
	select {
	case engine.taskQueue <- task:
		return nil
	case <-engine.shutdown:
		return fmt.Errorf("engine is shutdown")
	default:
		return fmt.Errorf("task queue is full")
	}
}

// GetResult 获取推理结果
func (engine *BatchInferenceEngine) GetResult() <-chan *InferenceResult {
	return engine.resultQueue
}

// Stop 停止引擎
func (engine *BatchInferenceEngine) Stop() {
	close(engine.shutdown)

	engine.wg.Wait()

	close(engine.taskQueue)
	close(engine.resultQueue)

	close(engine.sessionPool.sessions)
	for session := range engine.sessionPool.sessions {
		session.Destroy()
	}
}

// worker 工作协程
func (engine *BatchInferenceEngine) worker(id int) {
	defer engine.wg.Done()

	for {
		select {
		case task, ok := <-engine.taskQueue:
			if !ok {
				return
			}
			result := engine.processTask(task)
			if task.Callback != nil {
				select {
				case task.Callback <- result:
				case <-time.After(500 * time.Millisecond):
				}
			}
			select {
			case engine.resultQueue <- result:
			case <-time.After(500 * time.Millisecond):
			}
		case <-engine.shutdown:
			return
		}
	}
}

// processTask 处理单个推理任务
func (engine *BatchInferenceEngine) processTask(task *InferenceTask) *InferenceResult {
	session, err := engine.sessionPool.GetSession()
	if err != nil {
		return &InferenceResult{Error: fmt.Errorf("get session failed: %w", err)}
	}
	defer engine.sessionPool.PutSession(session)

	// 复制输入数据到张量
	copy(session.Input.GetData(), task.ImageData)

	// 运行推理
	err = session.Session.Run()
	if err != nil {
		return &InferenceResult{Error: fmt.Errorf("run inference failed: %w", err)}
	}

	// 处理输出
	// 注意：这里简化处理，实际需要根据模型输出格式进行解析
	// TODO: 需要使用 postprocess.go 中的 Postprocessor 来处理输出
	return &InferenceResult{Boxes: []BoundingBox{}}
}

// 辅助函数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 初始化ONNX Runtime环境
var (
	ortInitialized bool
	ortInitMutex   sync.Mutex
)

func initializeORTEnvironment() error {
	ortInitMutex.Lock()
	defer ortInitMutex.Unlock()
	if ortInitialized {
		return nil
	}
	libPath := getSharedLibPath()
	if libPath == "" {
		return errors.New("onnx runtime library not found")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize ort environment failed: %w", err)
	}
	ortInitialized = true
	return nil
}

// initSession 初始化ONNX Runtime会话
func initSession(modelPath string, inputSize, batchSize int) (*ModelSession, error) {
	if err := initializeORTEnvironment(); err != nil {
		return nil, err
	}

	inputShape := ort.NewShape(int64(batchSize), 3, int64(inputSize), int64(inputSize))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("create input tensor failed: %w", err)
	}

	outputShape := ort.NewShape(int64(batchSize), 84, 8400) // YOLO 输出
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor failed: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create session options failed: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create ort session failed: %w", err)
	}

	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

// getSharedLibPath 获取ONNX Runtime共享库路径
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	return ""
}

