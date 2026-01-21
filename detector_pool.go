package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// DetectionResult 检测结果
type DetectionResult struct {
	ImagePath string
	Objects   []boundingBox
	Error     error
	Metadata  map[string]interface{} // 额外元数据
}

// DetectionTask 检测任务
type DetectionTask struct {
	ImagePath string
	Callback  chan<- DetectionResult
	Timeout   time.Duration
}

// ModelSessionPool ONNX Runtime会话池
type ModelSessionPool struct {
	sessions  chan *ModelSession
	maxSize   int
	mutex     sync.Mutex
	modelPath string
}

// NewModelSessionPool 创建新的会话池
func NewModelSessionPool(maxSize int, modelPath string) *ModelSessionPool {
	return &ModelSessionPool{
		sessions:  make(chan *ModelSession, maxSize),
		maxSize:   maxSize,
		modelPath: modelPath,
	}
}

// GetSession 从池中获取会话，如果池为空则创建新会话
func (pool *ModelSessionPool) GetSession() (*ModelSession, error) {
	select {
	case session := <-pool.sessions:
		return session, nil
	default:
		// 池为空，创建新会话（注意：应限制总数避免资源耗尽）
		return pool.createSession()
	}
}

// PutSession 将会话放回池中
func (pool *ModelSessionPool) PutSession(session *ModelSession) {
	select {
	case pool.sessions <- session:
		// 成功放回池中
	default:
		// 池已满，销毁会话
		session.Destroy()
	}
}

// createSession 创建新的会话
func (pool *ModelSessionPool) createSession() (*ModelSession, error) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()

	// 检查当前池大小，避免无限增长
	if len(pool.sessions) >= pool.maxSize {
		return nil, fmt.Errorf("会话池已达到最大容量: %d", pool.maxSize)
	}

	session, err := initSession()
	if err != nil {
		return nil, err
	}

	return session, nil
}

// VideoDetectorManager 视频检测管理器
type VideoDetectorManager struct {
	taskQueue   chan *DetectionTask
	resultQueue chan DetectionResult
	sessionPool *ModelSessionPool
	workers     []*Worker
	workerCount int
	shutdown    chan struct{}
	wg          sync.WaitGroup
	timeout     time.Duration
}

// Worker 工作协程
type Worker struct {
	id       int
	manager  *VideoDetectorManager
	shutdown chan struct{}
}

// NewVideoDetectorManager 创建新的视频检测管理器
func NewVideoDetectorManager(workerCount, queueSize int, timeout time.Duration) *VideoDetectorManager {
	// 限制工作协程数量，最多不超过CPU核心数的2倍
	maxWorkers := runtime.NumCPU() * 2
	if workerCount > maxWorkers {
		fmt.Printf("警告: 工作协程数量 %d 超过推荐的最大值 %d，将限制为 %d\n", workerCount, maxWorkers, maxWorkers)
		workerCount = maxWorkers
	}

	maxSessions := workerCount
	if maxSessions > runtime.NumCPU()*2 {
		maxSessions = runtime.NumCPU() * 2 // 限制会话数量避免资源耗尽
	}

	manager := &VideoDetectorManager{
		taskQueue:   make(chan *DetectionTask, queueSize),
		resultQueue: make(chan DetectionResult, queueSize),
		sessionPool: NewModelSessionPool(maxSessions, modelPath),
		workers:     make([]*Worker, workerCount),
		workerCount: workerCount,
		shutdown:    make(chan struct{}),
		timeout:     timeout,
	}

	// 创建工作协程
	for i := 0; i < workerCount; i++ {
		worker := &Worker{
			id:       i,
			manager:  manager,
			shutdown: make(chan struct{}),
		}
		manager.workers[i] = worker
		manager.wg.Add(1)
		go worker.run()
	}

	return manager
}

// SubmitTask 提交检测任务
func (manager *VideoDetectorManager) SubmitTask(task *DetectionTask) error {
	select {
	case manager.taskQueue <- task:
		return nil
	case <-manager.shutdown:
		return fmt.Errorf("管理器已关闭")
	default:
		return fmt.Errorf("任务队列已满")
	}
}

// GetResult 获取检测结果
func (manager *VideoDetectorManager) GetResult() <-chan DetectionResult {
	return manager.resultQueue
}

// Stop 停止管理器
func (manager *VideoDetectorManager) Stop() {
	close(manager.shutdown)

	// 关闭所有工作协程
	for _, worker := range manager.workers {
		close(worker.shutdown)
	}

	// 等待所有工作协程结束
	manager.wg.Wait()

	// 关闭通道
	close(manager.taskQueue)
	close(manager.resultQueue)

	// 销毁会话池中的所有会话
	close(manager.sessionPool.sessions)
	for session := range manager.sessionPool.sessions {
		session.Destroy()
	}
}

// run 启动工作协程
func (worker *Worker) run() {
	defer worker.manager.wg.Done()

	for {
		select {
		case task, ok := <-worker.manager.taskQueue:
			if !ok {
				return
			}

			// 执行检测任务
			result := worker.processTask(task)

			// 发送结果
			select {
			case task.Callback <- result:
				// 通过回调发送结果
			case <-time.After(time.Second * 5): // 防止回调通道阻塞
				// 记录超时日志，但不阻塞工作协程
			}

			select {
			case worker.manager.resultQueue <- result:
				// 也发送到全局结果队列
			case <-time.After(time.Second * 5): // 防止结果队列阻塞
				// 记录超时日志，但不阻塞工作协程
			}

		case <-worker.shutdown:
			return
		}
	}
}

// processTask 处理单个检测任务
func (worker *Worker) processTask(task *DetectionTask) DetectionResult {
	// 从池中获取会话
	session, err := worker.manager.sessionPool.GetSession()
	if err != nil {
		return DetectionResult{
			ImagePath: task.ImagePath,
			Error:     fmt.Errorf("获取会话失败: %w", err),
		}
	}
	defer worker.manager.sessionPool.PutSession(session)

	// 加载图像
	originalPic, err := loadImageFile(task.ImagePath)
	if err != nil {
		return DetectionResult{
			ImagePath: task.ImagePath,
			Error:     fmt.Errorf("加载图像失败: %w", err),
		}
	}

	// 准备输入并运行推理
	scaleInfo, err := prepareInput(originalPic, session.Input)
	if err != nil {
		return DetectionResult{
			ImagePath: task.ImagePath,
			Error:     fmt.Errorf("准备输入失败: %w", err),
		}
	}

	err = session.Session.Run()
	if err != nil {
		return DetectionResult{
			ImagePath: task.ImagePath,
			Error:     fmt.Errorf("运行推理失败: %w", err),
		}
	}

	// 处理输出
	originalWidth := originalPic.Bounds().Dx()
	originalHeight := originalPic.Bounds().Dy()
	allBoxes := processOutput(session.Output.GetData(), originalWidth, originalHeight,
		float32(*confidenceThreshold), float32(*iouThreshold), scaleInfo)

	return DetectionResult{
		ImagePath: task.ImagePath,
		Objects:   allBoxes,
		Error:     nil,
		Metadata: map[string]interface{}{
			"timestamp": time.Now(),
			"worker_id": worker.id,
		},
	}
}

// ProcessImageBatch 批量处理图像的便捷方法
func (manager *VideoDetectorManager) ProcessImageBatch(imagePaths []string) []DetectionResult {
	results := make([]DetectionResult, len(imagePaths))
	callbacks := make([]chan DetectionResult, len(imagePaths))

	// 创建回调通道
	for i := range callbacks {
		callbacks[i] = make(chan DetectionResult, 1)
	}

	// 提交所有任务
	for i, imagePath := range imagePaths {
		task := &DetectionTask{
			ImagePath: imagePath,
			Callback:  callbacks[i],
		}

		err := manager.SubmitTask(task)
		if err != nil {
			results[i] = DetectionResult{
				ImagePath: imagePath,
				Error:     fmt.Errorf("提交任务失败: %w", err),
			}
		}
	}

	// 等待所有结果
	for i, callback := range callbacks {
		select {
		case result := <-callback:
			results[i] = result
		case <-time.After(manager.timeout):
			results[i] = DetectionResult{
				ImagePath: imagePaths[i],
				Error:     fmt.Errorf("处理超时"),
			}
		}
	}

	return results
}
