package main

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
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
	sessions       chan *ModelSession
	maxSize        int
	activeSessions int32 // 活跃会话计数，使用原子操作
	mutex          sync.Mutex
	modelPath      string
}

// NewModelSessionPool 创建新的会话池
func NewModelSessionPool(maxSize int, modelPath string) *ModelSessionPool {
	pool := &ModelSessionPool{
		sessions:  make(chan *ModelSession, maxSize),
		maxSize:   maxSize,
		modelPath: modelPath,
	}

	// 预创建一些会话，提高初始处理速度
	preCreateCount := max(1, min(maxSize/2, runtime.NumCPU()))
	for i := 0; i < preCreateCount; i++ {
		if session, err := initSession(); err == nil {
			select {
			case pool.sessions <- session:
			default:
				session.Destroy()
			}
		}
	}

	return pool
}

// GetSession 从池中获取会话，如果池为空则创建新会话
func (pool *ModelSessionPool) GetSession() (*ModelSession, error) {
	// 首先尝试从池中获取会话
	select {
	case session := <-pool.sessions:
		// 健康检查：验证会话是否有效
		if session != nil && session.Session != nil {
			atomic.AddInt32(&pool.activeSessions, 1)
			return session, nil
		}
		// 会话无效，销毁并继续尝试
		if session != nil {
			session.Destroy()
		}
	default:
	}

	// 池为空或会话无效，尝试创建新会话
	return pool.createSession()
}

// PutSession 将会话放回池中
func (pool *ModelSessionPool) PutSession(session *ModelSession) {
	// 减少活跃会话计数
	atomic.AddInt32(&pool.activeSessions, -1)

	// 检查会话是否有效
	if session == nil || session.Session == nil {
		return
	}

	// 将会话放回池中
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
	// 检查当前活跃会话数量，避免资源耗尽
	if atomic.LoadInt32(&pool.activeSessions) >= int32(pool.maxSize) {
		// 等待一段时间，看是否有会话被释放
		time.Sleep(10 * time.Millisecond)
		if atomic.LoadInt32(&pool.activeSessions) >= int32(pool.maxSize) {
			return nil, fmt.Errorf("活跃会话数量已达到最大容量: %d", pool.maxSize)
		}
	}

	// 创建新会话
	session, err := initSession()
	if err != nil {
		return nil, err
	}

	// 增加活跃会话计数
	atomic.AddInt32(&pool.activeSessions, 1)
	return session, nil
}

// GetStats 获取会话池统计信息
func (pool *ModelSessionPool) GetStats() (active, idle int) {
	active = int(atomic.LoadInt32(&pool.activeSessions))
	idle = len(pool.sessions)
	return
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

	// 根据系统内存调整队列大小，避免内存溢出
	systemMemory := runtime.MemStats{}
	runtime.ReadMemStats(&systemMemory)
	availableMemory := systemMemory.Sys - systemMemory.Alloc
	maxQueueSize := int(availableMemory / (1024 * 1024 * 10)) // 每10MB内存最多处理一个任务
	if queueSize > maxQueueSize && maxQueueSize > 0 {
		fmt.Printf("警告: 队列大小 %d 可能导致内存不足，将限制为 %d\n", queueSize, maxQueueSize)
		queueSize = maxQueueSize
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

	// 批量处理任务，减少上下文切换开销
	const batchSize = 4
	taskBatch := make([]*DetectionTask, 0, batchSize)

	for {
		// 尝试批量获取任务
		taskBatch = taskBatch[:0]
		batchTimeout := time.NewTimer(100 * time.Millisecond)

		// 最多等待100ms或直到收集到batchSize个任务
		for len(taskBatch) < batchSize {
			select {
			case task, ok := <-worker.manager.taskQueue:
				if !ok {
					batchTimeout.Stop()
					return
				}
				taskBatch = append(taskBatch, task)
			case <-batchTimeout.C:
				break
			case <-worker.shutdown:
				batchTimeout.Stop()
				return
			}
		}

		// 停止定时器
		batchTimeout.Stop()

		// 如果收集到了任务，批量处理
		if len(taskBatch) > 0 {
			for _, task := range taskBatch {
				// 执行检测任务
				result := worker.processTask(task)

				// 发送结果
				if task.Callback != nil {
					select {
					case task.Callback <- result:
						// 通过回调发送结果
					case <-time.After(500 * time.Millisecond): // 减少超时时间，提高响应速度
						// 记录超时日志，但不阻塞工作协程
					}
				}

				select {
				case worker.manager.resultQueue <- result:
					// 也发送到全局结果队列
				case <-time.After(500 * time.Millisecond): // 减少超时时间，提高响应速度
					// 记录超时日志，但不阻塞工作协程
				}
			}
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
