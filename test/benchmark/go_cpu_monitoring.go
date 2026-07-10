// go_cpu_monitoring.go
// Go CPU 使用率监控测试
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=8, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
// - 使用 gopsutil 库监控进程级 CPU 和内存
//
// 测试目的：
// - 测试四个场景的 CPU 使用率：空闲状态、单次推理、连续推理（100次）、并发推理（4 goroutine × 25次）
// - 记录 CPU 百分比、内存百分比、RSS
// - 为论文 CPU 利用率分析提供数据

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/process"
	ort "github.com/yalue/onnxruntime_go"
)

// CPUStats 记录CPU使用统计
type CPUStats struct {
	Timestamp     string  `json:"timestamp"`
	CPUPercent    float64 `json:"cpu_percent"`
	MemoryPercent float32 `json:"memory_percent"`
	RSSMB         float64 `json:"rss_mb"`
}

// TestResult 统一测试结果格式
type TestResult struct {
	TestName       string  `json:"test_name"`
	Model          string  `json:"model"`
	Language       string  `json:"language"`
	Scenario       string  `json:"scenario"`
	AvgCPUPercent  float64 `json:"avg_cpu_percent"`
	PeakCPUPercent float64 `json:"peak_cpu_percent"`
	StdCPUPercent  float64 `json:"std_cpu_percent"`
	AvgRSSMB       float64 `json:"avg_rss_mb"`
	PeakRSSMB      float64 `json:"peak_rss_mb"`
	Timestamp      string  `json:"timestamp"`
}

// CPUMonitor CPU监控器
type CPUMonitor struct {
	stats    []CPUStats
	stopChan chan bool
	wg       sync.WaitGroup
	process  *process.Process
}

// NewCPUMonitor 创建CPU监控器
func NewCPUMonitor() (*CPUMonitor, error) {
	pid := os.Getpid()
	p, err := process.NewProcess(int32(pid))
	if err != nil {
		return nil, fmt.Errorf("获取进程信息失败: %v", err)
	}
	return &CPUMonitor{
		stats:    make([]CPUStats, 0),
		stopChan: make(chan bool),
		process:  p,
	}, nil
}

// Start 开始监控
func (m *CPUMonitor) Start(interval time.Duration) {
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				cpuPercent, _ := m.process.CPUPercent()
				memPercent, _ := m.process.MemoryPercent()
				memInfo, _ := m.process.MemoryInfo()

				stat := CPUStats{
					Timestamp:     time.Now().Format("2006-01-02 15:04:05"),
					CPUPercent:    cpuPercent,
					MemoryPercent: memPercent,
					RSSMB:         float64(memInfo.RSS) / 1024 / 1024,
				}
				m.stats = append(m.stats, stat)

			case <-m.stopChan:
				return
			}
		}
	}()
}

// Stop 停止监控
func (m *CPUMonitor) Stop() {
	close(m.stopChan)
	m.wg.Wait()
}

// GetStats 获取统计结果
func (m *CPUMonitor) GetStats() (avgCPU, peakCPU, stdCPU float64, avgRSS, peakRSS float64) {
	if len(m.stats) == 0 {
		return 0, 0, 0, 0, 0
	}

	var sumCPU, sumRSS float64
	peakCPU = 0
	peakRSS = 0

	for _, stat := range m.stats {
		sumCPU += stat.CPUPercent
		sumRSS += stat.RSSMB
		if stat.CPUPercent > peakCPU {
			peakCPU = stat.CPUPercent
		}
		if stat.RSSMB > peakRSS {
			peakRSS = stat.RSSMB
		}
	}

	avgCPU = sumCPU / float64(len(m.stats))
	avgRSS = sumRSS / float64(len(m.stats))

	// 计算标准差
	var sumSquares float64
	for _, stat := range m.stats {
		diff := stat.CPUPercent - avgCPU
		sumSquares += diff * diff
	}
	stdCPU = math.Sqrt(sumSquares / float64(len(m.stats)))

	return avgCPU, peakCPU, stdCPU, avgRSS, peakRSS
}

// SaveToFile 保存监控数据到文件
func (m *CPUMonitor) SaveToFile(filename string) error {
	data, err := json.MarshalIndent(m.stats, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// 创建Session
func createSession(modelPath string, inputData []byte, inputShape, outputShape []int64) (*ort.AdvancedSession, *ort.Tensor[float32], *ort.Tensor[float32], error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("创建会话选项失败: %v", err)
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(8)
	opts.SetInterOpNumThreads(1)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("创建输入Tensor失败: %v", err)
	}

	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, nil, nil, fmt.Errorf("创建输出Tensor失败: %v", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		opts,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, nil, nil, fmt.Errorf("创建Session失败: %v", err)
	}

	return session, inputTensor, outputTensor, nil
}

// 场景1：空闲状态CPU监控
func testIdleCPU(basePath string) *TestResult {
	fmt.Println("\n===== 场景1：空闲状态CPU监控 =====")

	monitor, err := NewCPUMonitor()
	if err != nil {
		fmt.Printf("创建监控器失败: %v\n", err)
		return nil
	}

	fmt.Println("监控空闲状态 30 秒...")
	monitor.Start(1 * time.Second)
	time.Sleep(30 * time.Second)
	monitor.Stop()

	avgCPU, peakCPU, stdCPU, avgRSS, peakRSS := monitor.GetStats()

	fmt.Printf("平均CPU使用率: %.2f%%\n", avgCPU)
	fmt.Printf("峰值CPU使用率: %.2f%%\n", peakCPU)
	fmt.Printf("CPU使用率标准差: %.2f%%\n", stdCPU)
	fmt.Printf("平均RSS: %.2f MB\n", avgRSS)
	fmt.Printf("峰值RSS: %.2f MB\n", peakRSS)

	monitor.SaveToFile(filepath.Join(basePath, "results", "go_cpu_idle_stats.json"))

	return &TestResult{
		TestName:       "CPU_Monitoring_Idle",
		Model:          "None",
		Language:       "Go",
		Scenario:       "Idle",
		AvgCPUPercent:  avgCPU,
		PeakCPUPercent: peakCPU,
		StdCPUPercent:  stdCPU,
		AvgRSSMB:       avgRSS,
		PeakRSSMB:      peakRSS,
		Timestamp:      time.Now().Format("2006-01-02 15:04:05"),
	}
}

// 场景2：单次推理CPU监控
func testSingleInferenceCPU(modelPath, inputDataPath string, basePath string) *TestResult {
	fmt.Println("\n===== 场景2：单次推理CPU监控 =====")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return nil
	}

	session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
		[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
	if err != nil {
		fmt.Printf("创建Session失败: %v\n", err)
		return nil
	}
	defer session.Destroy()
	defer inputTensor.Destroy()
	defer outputTensor.Destroy()

	monitor, err := NewCPUMonitor()
	if err != nil {
		fmt.Printf("创建监控器失败: %v\n", err)
		return nil
	}

	fmt.Println("执行单次推理并监控CPU...")
	monitor.Start(100 * time.Millisecond)

	err = session.Run()
	if err != nil {
		fmt.Printf("推理失败: %v\n", err)
		monitor.Stop()
		return nil
	}

	time.Sleep(1 * time.Second) // 继续监控1秒
	monitor.Stop()

	avgCPU, peakCPU, stdCPU, avgRSS, peakRSS := monitor.GetStats()

	fmt.Printf("平均CPU使用率: %.2f%%\n", avgCPU)
	fmt.Printf("峰值CPU使用率: %.2f%%\n", peakCPU)
	fmt.Printf("CPU使用率标准差: %.2f%%\n", stdCPU)
	fmt.Printf("平均RSS: %.2f MB\n", avgRSS)
	fmt.Printf("峰值RSS: %.2f MB\n", peakRSS)

	monitor.SaveToFile(filepath.Join(basePath, "results", "go_cpu_single_inference_stats.json"))

	return &TestResult{
		TestName:       "CPU_Monitoring_Single_Inference",
		Model:          "YOLO11x",
		Language:       "Go",
		Scenario:       "Single_Inference",
		AvgCPUPercent:  avgCPU,
		PeakCPUPercent: peakCPU,
		StdCPUPercent:  stdCPU,
		AvgRSSMB:       avgRSS,
		PeakRSSMB:      peakRSS,
		Timestamp:      time.Now().Format("2006-01-02 15:04:05"),
	}
}

// 场景3：连续推理CPU监控
func testContinuousInferenceCPU(modelPath, inputDataPath string, basePath string) *TestResult {
	fmt.Println("\n===== 场景3：连续推理CPU监控（100次推理，约2分钟） =====")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return nil
	}

	session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
		[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
	if err != nil {
		fmt.Printf("创建Session失败: %v\n", err)
		return nil
	}
	defer session.Destroy()
	defer inputTensor.Destroy()
	defer outputTensor.Destroy()

	monitor, err := NewCPUMonitor()
	if err != nil {
		fmt.Printf("创建监控器失败: %v\n", err)
		return nil
	}

	fmt.Println("执行100次推理并监控CPU...")
	monitor.Start(1 * time.Second)

	for i := 0; i < 100; i++ {
		err = session.Run()
		if err != nil {
			fmt.Printf("第%d次推理失败: %v\n", i+1, err)
			break
		}
		if i%10 == 0 {
			fmt.Printf("已完成 %d/100 次推理\n", i)
		}
	}

	monitor.Stop()

	avgCPU, peakCPU, stdCPU, avgRSS, peakRSS := monitor.GetStats()

	fmt.Printf("平均CPU使用率: %.2f%%\n", avgCPU)
	fmt.Printf("峰值CPU使用率: %.2f%%\n", peakCPU)
	fmt.Printf("CPU使用率标准差: %.2f%%\n", stdCPU)
	fmt.Printf("平均RSS: %.2f MB\n", avgRSS)
	fmt.Printf("峰值RSS: %.2f MB\n", peakRSS)

	monitor.SaveToFile(filepath.Join(basePath, "results", "go_cpu_continuous_stats.json"))

	return &TestResult{
		TestName:       "CPU_Monitoring_Continuous",
		Model:          "YOLO11x",
		Language:       "Go",
		Scenario:       "Continuous_100_Inferences",
		AvgCPUPercent:  avgCPU,
		PeakCPUPercent: peakCPU,
		StdCPUPercent:  stdCPU,
		AvgRSSMB:       avgRSS,
		PeakRSSMB:      peakRSS,
		Timestamp:      time.Now().Format("2006-01-02 15:04:05"),
	}
}

// 场景4：并发推理CPU监控
func testConcurrentInferenceCPU(modelPath, inputDataPath string, basePath string) *TestResult {
	fmt.Println("\n===== 场景4：并发推理CPU监控（10并发，50次请求） =====")

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		return nil
	}

	monitor, err := NewCPUMonitor()
	if err != nil {
		fmt.Printf("创建监控器失败: %v\n", err)
		return nil
	}

	concurrency := 10
	numRequests := 50

	monitor.Start(500 * time.Millisecond)

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, concurrency)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(idx int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			session, inputTensor, outputTensor, err := createSession(modelPath, inputData,
				[]int64{1, 3, 640, 640}, []int64{1, 84, 8400})
			if err != nil {
				fmt.Printf("请求%d创建Session失败: %v\n", idx, err)
				return
			}
			defer session.Destroy()
			defer inputTensor.Destroy()
			defer outputTensor.Destroy()

			err = session.Run()
			if err != nil {
				fmt.Printf("请求%d推理失败: %v\n", idx, err)
			}

			if idx%10 == 0 {
				fmt.Printf("已完成 %d/%d 次请求\n", idx, numRequests)
			}
		}(i)
	}

	wg.Wait()
	monitor.Stop()

	avgCPU, peakCPU, stdCPU, avgRSS, peakRSS := monitor.GetStats()

	fmt.Printf("平均CPU使用率: %.2f%%\n", avgCPU)
	fmt.Printf("峰值CPU使用率: %.2f%%\n", peakCPU)
	fmt.Printf("CPU使用率标准差: %.2f%%\n", stdCPU)
	fmt.Printf("平均RSS: %.2f MB\n", avgRSS)
	fmt.Printf("峰值RSS: %.2f MB\n", peakRSS)

	monitor.SaveToFile(filepath.Join(basePath, "results", "go_cpu_concurrent_stats.json"))

	return &TestResult{
		TestName:       "CPU_Monitoring_Concurrent",
		Model:          "YOLO11x",
		Language:       "Go",
		Scenario:       "Concurrent_10x50",
		AvgCPUPercent:  avgCPU,
		PeakCPUPercent: peakCPU,
		StdCPUPercent:  stdCPU,
		AvgRSSMB:       avgRSS,
		PeakRSSMB:      peakRSS,
		Timestamp:      time.Now().Format("2006-01-02 15:04:05"),
	}
}

func main() {
	fmt.Println("===== Go CPU 使用率监控测试 =====")

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	runtime.GOMAXPROCS(12)

	results := make([]*TestResult, 0)

	// 场景1：空闲状态
	result1 := testIdleCPU(basePath)
	if result1 != nil {
		results = append(results, result1)
	}

	// 场景2：单次推理
	result2 := testSingleInferenceCPU(modelPath, inputDataPath, basePath)
	if result2 != nil {
		results = append(results, result2)
	}

	// 场景3：连续推理
	result3 := testContinuousInferenceCPU(modelPath, inputDataPath, basePath)
	if result3 != nil {
		results = append(results, result3)
	}

	// 场景4：并发推理
	result4 := testConcurrentInferenceCPU(modelPath, inputDataPath, basePath)
	if result4 != nil {
		results = append(results, result4)
	}

	// 保存所有结果
	resultData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("序列化结果失败: %v\n", err)
		os.Exit(1)
	}

	resultFile := filepath.Join(basePath, "results", "go_cpu_monitoring_result.json")
	err = os.WriteFile(resultFile, resultData, 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n===== 所有测试完成 =====\n")
	fmt.Printf("结果已保存到: %s\n", resultFile)

	// 打印汇总
	fmt.Println("\n===== CPU 监控汇总 =====")
	for _, r := range results {
		fmt.Printf("\n场景: %s\n", r.Scenario)
		fmt.Printf("  平均CPU: %.2f%% | 峰值CPU: %.2f%% | 平均RSS: %.2f MB\n",
			r.AvgCPUPercent, r.PeakCPUPercent, r.AvgRSSMB)
	}
}
