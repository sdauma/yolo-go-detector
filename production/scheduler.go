package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Scheduler 轮巡调度器
// 每轮：并发取图 → Session Pool 推理 → 输出结果
// 轮次非阻塞：下一轮按 interval 启动，不等待当前轮完成（最多 3 轮重叠）
type Scheduler struct {
	cfg      *Config
	camMgr   *CameraManager
	pipeline *DetectionPipeline
	output   *OutputWriter
	logger   *Logger

	ctx    context.Context
	cancel context.CancelFunc

	// 非阻塞轮次控制：限制同时运行的轮次数量
	runningRounds chan struct{}
}

// NewScheduler 创建调度器
func NewScheduler(cfg *Config, camMgr *CameraManager, pipeline *DetectionPipeline, output *OutputWriter, logger *Logger) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	return &Scheduler{
		cfg:           cfg,
		camMgr:        camMgr,
		pipeline:      pipeline,
		output:        output,
		logger:        logger,
		ctx:           ctx,
		cancel:        cancel,
		runningRounds: make(chan struct{}, 3), // 最多 3 轮同时运行
	}
}

// Run 启动调度循环
func (s *Scheduler) Run() {
	// 第一步：等待登录就绪（首次登录可能耗时较长，chromedp 登录需数十秒）
	s.logger.Printf("[调度] 等待登录就绪...\n")
	if err := s.camMgr.Login(s.ctx); err != nil {
		s.logger.Printf("[调度] 登录失败，无法启动调度: %v\n", err)
		return
	}
	s.logger.Printf("[调度] 登录就绪，开始获取在线摄像头列表...\n")

	// 第二步：登录后刷新在线列表
	if err := s.camMgr.RefreshOnlineList(); err != nil {
		s.logger.Printf("[调度] 首次刷新在线列表失败: %v\n", err)
		// 不退出，继续跑调度（可能是临时网络问题）
	}

	roundInterval := time.Duration(s.cfg.Scheduler.RoundIntervalSeconds) * time.Second
	refreshInterval := time.Duration(s.cfg.VideoAPI.RefreshIntervalMinutes) * time.Minute

	ticker := time.NewTicker(roundInterval)
	refreshTicker := time.NewTicker(refreshInterval)

	roundCount := 0

	s.logger.Printf("[调度] 启动成功，每 %.0f 秒一轮，每 %.0f 分钟刷新在线列表，当前在线 %d 路\n",
		roundInterval.Seconds(), refreshInterval.Minutes(), s.camMgr.OnlineCount())

	for {
		select {
		case <-s.ctx.Done():
			ticker.Stop()
			refreshTicker.Stop()
			s.logger.Printf("[调度] 已停止\n")
			return

		case <-refreshTicker.C:
			s.logger.Printf("[调度] 定时刷新在线摄像头列表...\n")
			if err := s.camMgr.RefreshOnlineList(); err != nil {
				s.logger.Printf("[调度] 刷新在线列表失败: %v\n", err)
			} else {
				s.logger.Printf("[调度] 在线列表刷新完成，当前在线 %d 路\n", s.camMgr.OnlineCount())
			}

		case <-ticker.C:
			roundCount++
			// 非阻塞：在当前 goroutine 获取信号量后，在后台 goroutine 执行本轮
			// 这样 ticker 不会因为 round 耗时过长而错过间隔
			select {
			case s.runningRounds <- struct{}{}:
				go func(roundNum int) {
					defer func() { <-s.runningRounds }()
					s.runRound(roundNum)
				}(roundCount)
			default:
				s.logger.Printf("[调度] 已有 %d 轮正在运行，跳过第 %d 轮\n", cap(s.runningRounds), roundCount)
			}
		}
	}
}

// runRound 执行一轮检测
func (s *Scheduler) runRound(roundNum int) {
	cameras := s.camMgr.GetOnlineCameras()
	if len(cameras) == 0 {
		s.logger.Printf("[第 %d 轮] 没有在线摄像头，跳过\n", roundNum)
		return
	}

	t0 := time.Now()
	s.logger.Printf("[第 %d 轮] 开始检测 %d 路摄像头...\n", roundNum, len(cameras))

	// 每路取图超时（用于子请求）
	fetchTimeout := time.Duration(s.cfg.Scheduler.FetchTimeoutSeconds) * time.Second
	if fetchTimeout <= 0 {
		fetchTimeout = 30 * time.Second
	}

	// 整轮超时 = 预估时间 * 安全系数
	// 预估时间 = (摄像头数/并发数) * 单路超时
	totalCameras := len(cameras)
	concurrency := s.cfg.Scheduler.FetchConcurrency
	if concurrency <= 0 {
		concurrency = 100
	}
	estimatedTime := time.Duration(totalCameras/concurrency+1) * fetchTimeout
	roundDeadline := t0.Add(estimatedTime * 2) // 2倍安全系数
	roundCtx, roundCancel := context.WithDeadline(s.ctx, roundDeadline)
	defer roundCancel()

	// 并发取图 + 检测
	type taskResult struct {
		cam     *CameraInfo
		imgData []byte
		detect  *DetectResult
	}

	results := make([]*taskResult, totalCameras)
	var completed int32 // atomic counter for progress

	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, cam := range cameras {
		wg.Add(1)
		go func(idx int, c *CameraInfo) {
			defer wg.Done()

			// 循环检查：等待信号量时也响应取消
			for {
				select {
				case <-roundCtx.Done():
					results[idx] = &taskResult{
						cam: c,
						detect: &DetectResult{
							ChannelCode: c.ChannelCode,
							ChannelName: c.ChannelName,
							OrgName:     c.OrgName,
							Timestamp:   time.Now().Format("2006-01-02 15:04:05"),
							Error:       fmt.Sprintf("cancelled: %v", roundCtx.Err()),
						},
					}
					return
				case sem <- struct{}{}:
					goto gotSemaphore
				}
			}
		gotSemaphore:
			defer func() { <-sem }()

			tr := &taskResult{cam: c}

			// 1. 取图（context 控制超时）
			imgData, err := s.camMgr.FetchSnapshot(roundCtx, c.ChannelCode)
			if err != nil {
				tr.detect = &DetectResult{
					ChannelCode: c.ChannelCode,
					ChannelName: c.ChannelName,
					OrgName:     c.OrgName,
					Timestamp:   time.Now().Format("2006-01-02 15:04:05"),
					Error:       fmt.Sprintf("fetch: %v", err),
				}
				results[idx] = tr
				atomic.AddInt32(&completed, 1)
				return
			}
			tr.imgData = imgData

			// 检查是否在取图过程中被取消
			select {
			case <-roundCtx.Done():
				tr.detect = &DetectResult{
					ChannelCode: c.ChannelCode,
					ChannelName: c.ChannelName,
					OrgName:     c.OrgName,
					Timestamp:   time.Now().Format("2006-01-02 15:04:05"),
					Error:       fmt.Sprintf("cancelled after fetch: %v", roundCtx.Err()),
				}
				results[idx] = tr
				atomic.AddInt32(&completed, 1)
				return
			default:
			}

			// 2. 检测
			detectResult := s.pipeline.Detect(c, imgData)
			tr.detect = detectResult
			results[idx] = tr
			atomic.AddInt32(&completed, 1)
		}(i, cam)
	}

	// 进度日志 goroutine：每 5 秒输出一次进度
	progressDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				done := atomic.LoadInt32(&completed)
				s.logger.Printf("[第 %d 轮] 进度: %d/%d (%.0f%%), 已耗时 %ds\n",
					roundNum, done, totalCameras,
					float64(done)/float64(totalCameras)*100,
					int(time.Since(t0).Seconds()))
			case <-progressDone:
				return
			}
		}
	}()

	wg.Wait()
	close(progressDone)

	// 统计 + 输出
	var (
		successCount int
		failCount    int
		alertCount   int
		totalFetchMs int64
		totalInferMs int64
	)

	for _, tr := range results {
		if tr == nil || tr.detect == nil {
			continue
		}

		if tr.detect.Error != "" {
			failCount++
		} else {
			successCount++
		}

		if s.pipeline.HasAlert(tr.detect) {
			alertCount++

			// 告警图片存盘
			if tr.imgData != nil {
				path, err := s.output.SaveAlertImage(tr.detect, tr.imgData, s.pipeline)
				if err != nil {
					s.logger.Printf("  [告警] 保存图片失败 [%s]: %v\n", tr.cam.ChannelCode, err)
				} else if path != "" {
					s.logger.Printf("  [告警] %s: %d 个目标 → %s\n",
						tr.cam.ChannelName, len(tr.detect.Detections), path)
				}
			}
		}

		// 写入 JSONL
		if err := s.output.WriteResult(tr.detect); err != nil {
			s.logger.Printf("  [输出] 写入 JSONL 失败: %v\n", err)
		}

		totalFetchMs += tr.detect.FetchMs
		totalInferMs += tr.detect.InferMs
	}

	roundMs := time.Since(t0).Milliseconds()
	active, idle := s.pipeline.PoolStats()

	avgFetchMs := int64(0)
	avgInferMs := int64(0)
	if successCount > 0 {
		avgFetchMs = totalFetchMs / int64(successCount)
		avgInferMs = totalInferMs / int64(successCount)
	}

	// 写入本轮摘要
	summary := &RoundSummary{
		Type:         "round_summary",
		Timestamp:    time.Now().Format("2006-01-02 15:04:05"),
		TotalOnline:  len(cameras),
		SuccessCount: successCount,
		FailCount:    failCount,
		AlertCount:   alertCount,
		RoundTotalMs: roundMs,
		AvgFetchMs:   avgFetchMs,
		AvgInferMs:   avgInferMs,
		PoolActive:   active,
		PoolIdle:     idle,
	}
	s.output.WriteRoundSummary(summary)

	s.logger.Printf("[第 %d 轮] 完成: %d/%d 成功, %d 告警, 取图均 %dms, 推理均 %dms, 总耗时 %dms, Pool(%d/%d)\n",
		roundNum, successCount, len(cameras), alertCount,
		avgFetchMs, avgInferMs, roundMs, active, active+idle)
}

// Stop 停止调度器
func (s *Scheduler) Stop() {
	s.cancel()
	s.output.Close()
}
