package engine

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
	intraOpThreads int // 每 Session 的 intra_op 线程数
}

// NewSessionPool 创建新的会话池
// intraOpThreads: 每 Session 内部并行线程数，0=自动计算（CPU核数/poolSize，最小1）
func NewSessionPool(maxSize int, modelPath string, inputSize, batchSize int, intraOpThreads int) *SessionPool {
	if intraOpThreads <= 0 {
		intraOpThreads = max(1, runtime.NumCPU()/maxSize)
	}

	pool := &SessionPool{
		sessions:       make(chan *ModelSession, maxSize),
		maxSize:        maxSize,
		modelPath:      modelPath,
		inputSize:      inputSize,
		batchSize:      batchSize,
		intraOpThreads: intraOpThreads,
	}

	// 预创建全部会话，避免运行时创建开销
	for i := 0; i < maxSize; i++ {
		if session, err := createSessionInternal(modelPath, inputSize, batchSize, intraOpThreads); err == nil {
			pool.sessions <- session
		} else {
			fmt.Printf("[SessionPool] 警告: 预创建第 %d 个 Session 失败: %v\n", i+1, err)
		}
	}
	fmt.Printf("[SessionPool] 预创建 %d 个 Session（池容量 %d，每 Session %d 线程）\n",
		len(pool.sessions), maxSize, intraOpThreads)

	return pool
}

// GetSession 从池中获取会话（阻塞等待，不会超过池容量）
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

	// 池为空，检查是否可以创建新会话
	currentActive := atomic.LoadInt32(&pool.activeSessions)
	if currentActive < int32(pool.maxSize) {
		// 使用 CAS 操作确保不会超过最大值
		if atomic.CompareAndSwapInt32(&pool.activeSessions, currentActive, currentActive+1) {
			return pool.createSession()
		}
		// CAS 失败，说明有其他 goroutine 抢先创建了会话，重试获取
		return pool.GetSession()
	}

	// ★ 关键修复：池已满，阻塞等待（之前是返回 error 导致丢失检测）
	// 生产环境中 30 路摄像头并发时需要排队等待
	session := <-pool.sessions
	if session != nil && session.Session != nil {
		atomic.AddInt32(&pool.activeSessions, 1)
		return session, nil
	}
	if session != nil {
		session.Destroy()
	}
	return pool.GetSession()
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

// createSession 创建新的会话（内部使用，使用与预创建一致的线程配置）
func (pool *SessionPool) createSession() (*ModelSession, error) {
	session, err := initSessionWithThreads(pool.modelPath, pool.inputSize, pool.batchSize, pool.intraOpThreads)
	if err != nil {
		atomic.AddInt32(&pool.activeSessions, -1)
		return nil, err
	}

	return session, nil
}

// createSessionInternal 创建会话（用于预创建，绕过池的 createSession 计数器逻辑）
func createSessionInternal(modelPath string, inputSize, batchSize, intraOpThreads int) (*ModelSession, error) {
	return initSessionWithThreads(modelPath, inputSize, batchSize, intraOpThreads)
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
		sessionPool: NewSessionPool(maxSessions, modelPath, inputSize, batchSize, 0),
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
	// 注意：BatchInferenceEngine 为旧版异步回调 API，生产环境已改用
	// Postprocessor + SessionPool 的同步推理（见 production/detector.go）。
	// 此处返回空 Boxes 是有意为之——完整后处理见 Postprocessor。
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
	ortInitialized      bool
	ortInitMutex        sync.Mutex
	configuredONNXLibPath string // 由外部通过 SetONNXLibPath() 设置
)

// SetONNXLibPath 设置 ONNX Runtime 动态库路径（从配置文件传入）
// 在调用任何推理功能之前调用，空字符串表示使用自动搜索
func SetONNXLibPath(path string) {
	configuredONNXLibPath = path
}

func initializeORTEnvironment() error {
	ortInitMutex.Lock()
	defer ortInitMutex.Unlock()
	if ortInitialized {
		return nil
	}
	libPath := getSharedLibPath()
	if libPath == "" {
		return errors.New("onnx runtime library not found, set onnx_lib_path in config or ensure third_party/onnxruntime.dll exists")
	}
	if _, err := os.Stat(libPath); err != nil {
		return fmt.Errorf("ONNX Runtime 动态库不存在: %s (请检查配置 onnx_lib_path 或确认 %s 目录下有对应文件)", libPath, filepath.Dir(libPath))
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize ort environment failed: %w (库路径: %s)", err, libPath)
	}
	ortInitialized = true
	return nil
}

// initSession 初始化ONNX Runtime会话（默认线程数 = CPU 核数，适用于单 Session 场景）
func initSession(modelPath string, inputSize, batchSize int) (*ModelSession, error) {
	return initSessionWithThreads(modelPath, inputSize, batchSize, 0)
}

// initSessionWithThreads 初始化会话并显式设置线程数
// intraOpThreads: 0=自动（CPU核数），其他=显式设置
func initSessionWithThreads(modelPath string, inputSize, batchSize, intraOpThreads int) (*ModelSession, error) {
	if err := initializeORTEnvironment(); err != nil {
		return nil, err
	}

	if intraOpThreads <= 0 {
		intraOpThreads = runtime.NumCPU()
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

	// ★ 关键修复：显式设置线程数，避免 CPU 过度订阅
	// 未设置时 ONNX Runtime 默认用满全部 CPU 核心，多 Session 并发时造成灾难性线程争抢
	options.SetIntraOpNumThreads(intraOpThreads)
	options.SetInterOpNumThreads(1)

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
// 优先级: 1) 配置文件指定路径  2) exe 同级 third_party/  3) 向上搜索父目录
func getSharedLibPath() string {
	// 0. 如果外部通过 SetONNXLibPath 设置了路径，优先使用
	if configuredONNXLibPath != "" {
		if _, err := os.Stat(configuredONNXLibPath); err == nil {
			return configuredONNXLibPath
		}
		// 配置的路径不存在，打印告警并继续搜索
		fmt.Printf("[WARNING] 配置的 onnx_lib_path 不存在: %s，回退到自动搜索\n", configuredONNXLibPath)
	}

	// 获取当前可执行文件所在目录
	exePath, err := os.Executable()
	if err != nil {
		// 如果获取失败，使用当前工作目录
		exePath = "."
	}
	basePath := filepath.Dir(exePath)

	// 确定各平台的库文件名
	var libFileName string
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		libFileName = "onnxruntime.dll"
	} else if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		libFileName = "onnxruntime_arm64.dylib"
	} else if runtime.GOOS == "darwin" && runtime.GOARCH == "amd64" {
		libFileName = "onnxruntime_amd64.dylib"
	} else if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		libFileName = "onnxruntime_arm64.so"
	} else if runtime.GOOS == "linux" {
		libFileName = "onnxruntime.so"
	} else {
		return ""
	}

	// 从 exe 目录向上搜索，最多搜 5 层父目录
	searchPath := basePath
	for i := 0; i < 5; i++ {
		candidate := filepath.Join(searchPath, "third_party", libFileName)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(searchPath)
		if parent == searchPath {
			break // 到达根目录
		}
		searchPath = parent
	}

	// 回退：返回 exe 目录下的路径，由调用方检查文件是否存在并打印告警
	return filepath.Join(basePath, "third_party", libFileName)
}

