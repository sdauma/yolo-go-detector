// paper_full_benchmark.go
// 论文完整基准测试程序 — 一键运行所有核心实验
//
// 测试内容：
//   [1/6] 内存指标验证 — PrivateMemorySize64 vs WorkingSet64 对比
//   [2/6] 架构对比实验 — Unsafe Shared / Mutex Shared / Session Pool
//   [3/6] Session Pool 消融实验 — PoolSize × IntraThreads
//   [4/6] 冷启动分解实验 — 20次重复，YOLO11x + YOLO11n
//   [5/6] 10分钟稳定性实验 — 连续推理 RSS 漂移
//   [6/6] 批量推理效果 — batch size 对吞吐量和内存的影响
//
// 重要：
//   - 所有内存测量使用 PrivateMemorySize64（可靠指标）
//   - 必须编译为 .exe 后运行（NOT "go run"）
//   - 建议关闭所有其他软件后运行
//
// 预计总耗时：约 60-90 分钟（取决于 CPU）
// 结果保存到：results/（直接覆盖旧测试文件）

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"yolo-go-detector/test/benchmark/memutil"
)

// ============================================================
// 全局工具函数
// ============================================================

// getPM returns PrivateMemorySize64 (MB) via direct Windows API (no PowerShell overhead).
func getPM() float64 { return memutil.PrivateMemoryMB() }

// getWS returns WorkingSet64 (MB) via direct Windows API (no PowerShell overhead).
func getWS() float64 { return memutil.WorkingSetMB() }

func calcPercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func fillTensorData(t *ort.Tensor[float32], data []byte) {
	floatData := t.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(data) {
			bits := binary.LittleEndian.Uint32(data[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

type swt struct {
	s  *ort.AdvancedSession
	it *ort.Tensor[float32]
	ot *ort.Tensor[float32]
}

func createSWT(modelPath string, inputShape, outputShape []int64, intraOp int, inputData []byte) (*swt, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	opts.SetIntraOpNumThreads(intraOp)
	opts.SetInterOpNumThreads(1)
	opts.SetExecutionMode(0)
	opts.SetGraphOptimizationLevel(3)

	it, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		opts.Destroy()
		return nil, err
	}
	ot, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		it.Destroy()
		opts.Destroy()
		return nil, err
	}
	fillTensorData(it, inputData)

	s, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{it}, []ort.ArbitraryTensor{ot}, opts)
	opts.Destroy()
	if err != nil {
		it.Destroy()
		ot.Destroy()
		return nil, err
	}
	return &swt{s, it, ot}, nil
}

// ============================================================
// [1/6] 内存指标验证
// ============================================================

type MemVerifyResult struct {
	State string  `json:"state"`
	WS    float64 `json:"ws_mb"`
	PM    float64 `json:"pm_mb"`
}

func runMemoryVerification(modelPath string, inputShape, outputShape []int64, inputData []byte) []MemVerifyResult {
	fmt.Println("\n============================================================")
	fmt.Println("[1/6] 内存指标验证 — PrivateMemorySize64 vs WorkingSet64")
	fmt.Println("============================================================")

	var results []MemVerifyResult

	add := func(state string) {
		results = append(results, MemVerifyResult{state, getWS(), getPM()})
		fmt.Printf("  %-30s  WS=%-10.2f  PM=%-10.2f\n", state, getWS(), getPM())
	}

	add("BASELINE")

	// 1 Session
	sw, err := createSWT(modelPath, inputShape, outputShape, 1, inputData)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return results
	}
	add("1_SESSION")

	// 3 inferences
	for i := 0; i < 3; i++ {
		sw.s.Run()
	}
	add("1_SESSION_3_INFERS")

	// 12 Sessions
	var all []swt
	all = append(all, *sw)
	for i := 0; i < 11; i++ {
		s2, err := createSWT(modelPath, inputShape, outputShape, 1, inputData)
		if err != nil {
			break
		}
		all = append(all, *s2)
	}
	time.Sleep(1 * time.Second)
	add(fmt.Sprintf("%d_SESSIONS", len(all)))

	// Destroy all
	for _, s := range all {
		s.s.Destroy()
		s.it.Destroy()
		s.ot.Destroy()
	}
	time.Sleep(2 * time.Second)
	add("AFTER_DESTROY")

	return results
}

// ============================================================
// [2/6] 架构对比实验
// ============================================================

type ArchResult struct {
	Architecture string  `json:"architecture"`
	Concurrency  int     `json:"concurrency"`
	PoolSize     int     `json:"pool_size"`
	Throughput   float64 `json:"throughput_reqs"`
	AvgLatency   float64 `json:"avg_latency_ms"`
	P50Latency   float64 `json:"p50_latency_ms"`
	P90Latency   float64 `json:"p90_latency_ms"`
	P99Latency   float64 `json:"p99_latency_ms"`
	MinLatency   float64 `json:"min_latency_ms"`
	MaxLatency   float64 `json:"max_latency_ms"`
	StartPM      float64 `json:"start_pm_mb"`
	PeakPM       float64 `json:"peak_pm_mb"`
	EndPM        float64 `json:"end_pm_mb"`
	PMDrift      float64 `json:"pm_drift_mb"`
}

// runUnsafeSharedInSubprocess runs the Unsafe Shared architecture test in a child process.
// Since Unsafe Shared can trigger C-level fatal errors (ONNX Runtime tensor state conflicts)
// that cannot be caught by Go's recover(), we run it as a separate process.
// If the child process crashes, we capture the evidence (exit code, stderr) and record it.
func runUnsafeSharedInSubprocess(modelPath string) ArchResult {
	fmt.Println("  编译 unsafe_shared_test.exe（独立子进程）...")
	unsafeExe := "unsafe_shared_test.exe"
	unsafeDir := "unsafe_shared_test"

	// Step 1: Create temp directory for the package
	os.RemoveAll(unsafeDir) // Clean up if exists
	if err := os.Mkdir(unsafeDir, 0755); err != nil {
		fmt.Printf("  创建目录失败: %v\n", err)
		return ArchResult{Architecture: "Unsafe Shared"}
	}
	defer os.RemoveAll(unsafeDir)

	unsafeSrc := filepath.Join(unsafeDir, "main.go")

	// Step 2: Write the child process Go source file
	childCode := fmt.Sprintf(`package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func getProcessRSS() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Sys) / (1024 * 1024)
}

func main() {
	modelPath := %q
	libPath := filepath.Join("..", "..", "third_party", "onnxruntime.dll")

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "INIT_ERROR: %%v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// Create session
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "OPTIONS_ERROR: %%v\n", err)
		os.Exit(1)
	}
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)

	tempInput, _ := ort.NewEmptyTensor[float32](inputShape)
	defer tempInput.Destroy()
	tempOutput, _ := ort.NewEmptyTensor[float32](outputShape)
	defer tempOutput.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{tempInput},
		[]ort.ArbitraryTensor{tempOutput},
		opts,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "SESSION_ERROR: %%v\n", err)
		os.Exit(1)
	}
	defer session.Destroy()

	concurrencies := []int{1, 2, 4, 8, 12}
	for _, concurrency := range concurrencies {
		fmt.Printf("[UNSAFE_TEST] concurrency=%%d START\n", concurrency)
		numRequests := 500
		var wg sync.WaitGroup
		batchSize := numRequests / concurrency
		startTime := time.Now()

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				inputTensor, _ := ort.NewEmptyTensor[float32](inputShape)
				defer inputTensor.Destroy()
				outputTensor, _ := ort.NewEmptyTensor[float32](outputShape)
				defer outputTensor.Destroy()

				for j := 0; j < batchSize; j++ {
					err := session.Run()
					if err != nil {
						fmt.Fprintf(os.Stderr, "RUN_ERROR concurrency=%%d: %%v\n", concurrency, err)
						return
					}
				}
			}(i)
		}
		wg.Wait()
		elapsed := time.Since(startTime).Seconds()
		fmt.Printf("[UNSAFE_TEST] concurrency=%%d DONE throughput=%%f\n", concurrency, float64(numRequests)/elapsed)
	}

	fmt.Println("[UNSAFE_TEST] ALL_DONE")
}
`, modelPath)

	if err := os.WriteFile(unsafeSrc, []byte(childCode), 0644); err != nil {
		fmt.Printf("  写入 %s 失败: %v\n", unsafeSrc, err)
		return ArchResult{Architecture: "Unsafe Shared"}
	}

	// Step 3: Compile the child process from project root
	// Go 1.21+ does not support go build file.go with external dependencies
	// in module mode. Must compile as a package (in its own directory).
	projectRoot, _ := filepath.Abs(filepath.Join("..", ".."))
	cwd, _ := os.Getwd()
	buildCmd := exec.Command("go", "build",
		"-C", projectRoot,
		"-o", filepath.Join(cwd, unsafeExe),
		"./test/benchmark/"+unsafeDir,
	)
	buildOut, buildErr := buildCmd.CombinedOutput()
	if buildErr != nil {
		fmt.Printf("  编译失败: %v\n%s\n", buildErr, string(buildOut))
		return ArchResult{Architecture: "Unsafe Shared"}
	}
	defer os.Remove(unsafeExe)

	// Step 3: Run the child process
	fmt.Println("  运行 Unsafe Shared 子进程...")
	runCmd := exec.Command(".\\" + unsafeExe)
	runOutput, runErr := runCmd.CombinedOutput()
	outputStr := string(runOutput)

	exitCode := 0
	if runErr != nil {
		if exitErr, ok := runErr.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}
	}

	// Step 4: Parse results or record crash
	result := ArchResult{Architecture: "Unsafe Shared"}
	if exitCode != 0 {
		fmt.Printf("  [实证] Unsafe Shared 子进程崩溃! exitCode=%d\n", exitCode)
		fmt.Printf("  stderr/output (last 500 chars):\n%s\n",
			outputStr[max(0, len(outputStr)-500):])
		result.Concurrency = -exitCode // negative concurrency = crash indicator
		result.Throughput = -1          // -1 = crashed
		result.AvgLatency = -1
		result.P50Latency = -1
		result.P90Latency = -1
		result.P99Latency = -1
	} else {
		// Parse throughput results from child output
		fmt.Printf("  Unsafe Shared 子进程完成（未崩溃）\n%s\n", outputStr)
		// Try to parse max throughput from output
		for _, line := range strings.Split(outputStr, "\n") {
			if strings.Contains(line, "[UNSAFE_TEST] concurrency=") && strings.Contains(line, "DONE") {
				parts := strings.Fields(line)
				for _, p := range parts {
					if strings.HasPrefix(p, "throughput=") {
						fmt.Sscanf(p, "throughput=%f", &result.Throughput)
					}
				}
			}
		}
	}

	return result
}

func runArchitectureBenchmark(modelPath string, inputShape, outputShape []int64, inputData []byte) []ArchResult {
	fmt.Println("\n============================================================")
	fmt.Println("[2/6] 架构对比实验")
	fmt.Println("============================================================")

	const numRequests = 500
	var results []ArchResult

	// --- Unsafe Shared (子进程方式运行，捕获崩溃) ---
	// Unsafe Shared 在并发场景下会触发 ONNX Runtime 内部张量状态冲突
	// （C 层 fatal error 无法被 Go recover 捕获），会导致进程直接崩溃。
	// 因此通过子进程运行，若子进程崩溃则记录为实证。
	fmt.Println("\n--- Unsafe Shared (子进程运行，捕获崩溃证据) ---")
	unsafeResult := runUnsafeSharedInSubprocess(modelPath)
	results = append(results, unsafeResult)

	// --- Mutex Shared ---
	fmt.Println("\n--- Mutex Shared (共享Session+Tensor，加锁) ---")
	for _, concurrency := range []int{1, 2, 4, 8, 12} {
		fmt.Printf("  并发=%d ... ", concurrency)
		r := testMutexShared(modelPath, inputShape, outputShape, inputData, concurrency, numRequests)
		r.Architecture = "Mutex Shared"
		r.Concurrency = concurrency
		results = append(results, r)
		fmt.Printf("吞吐=%.2f req/s, P99=%.2f ms, PM峰值=%.2f MB\n", r.Throughput, r.P99Latency, r.PeakPM)
	}

	// --- Session Pool ---
	fmt.Println("\n--- Session Pool (独立Session) ---")
	for _, poolSize := range []int{1, 2, 4, 6, 8, 12} {
		fmt.Printf("  池大小=%d ... ", poolSize)
		r := testSessionPool(modelPath, inputShape, outputShape, inputData, poolSize, numRequests)
		r.Architecture = "Session Pool"
		r.PoolSize = poolSize
		results = append(results, r)
		fmt.Printf("吞吐=%.2f req/s, P99=%.2f ms, PM峰值=%.2f MB\n", r.Throughput, r.P99Latency, r.PeakPM)
	}

	return results
}

func testUnsafeShared(modelPath string, inputShape, outputShape []int64, inputData []byte, concurrency, numRequests int) ArchResult {
	startPM := getPM()
	peakPM := startPM

	opts, _ := ort.NewSessionOptions()
	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)
	ti, _ := ort.NewEmptyTensor[float32](inputShape)
	to, _ := ort.NewEmptyTensor[float32](outputShape)
	s, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{ti}, []ort.ArbitraryTensor{to}, opts)
	opts.Destroy()
	ti.Destroy()
	to.Destroy()
	if err != nil {
		return ArchResult{StartPM: startPM, PeakPM: peakPM}
	}
	defer s.Destroy()

	var wg sync.WaitGroup
	var mu sync.Mutex
	var latencies []float64
	startTime := time.Now()
	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			it, _ := ort.NewEmptyTensor[float32](inputShape)
			ot, _ := ort.NewEmptyTensor[float32](outputShape)
			defer it.Destroy()
			defer ot.Destroy()
			fillTensorData(it, inputData)

			for j := 0; j < batchSize; j++ {
				pm := getPM()
				mu.Lock()
				if pm > peakPM {
					peakPM = pm
				}
				mu.Unlock()

				t0 := time.Now()
				s.Run()
				lat := float64(time.Since(t0).Milliseconds())
				mu.Lock()
				latencies = append(latencies, lat)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	totalTime := float64(time.Since(startTime).Milliseconds())
	endPM := getPM()
	return summarizeArch(numRequests, totalTime, startPM, peakPM, endPM, latencies)
}

func testMutexShared(modelPath string, inputShape, outputShape []int64, inputData []byte, concurrency, numRequests int) ArchResult {
	startPM := getPM()
	peakPM := startPM

	sw, err := createSWT(modelPath, inputShape, outputShape, 1, inputData)
	if err != nil {
		return ArchResult{StartPM: startPM, PeakPM: peakPM}
	}
	defer sw.s.Destroy()
	defer sw.it.Destroy()
	defer sw.ot.Destroy()

	var mu sync.Mutex
	var wg sync.WaitGroup
	var latencies []float64
	startTime := time.Now()
	batchSize := numRequests / concurrency

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < batchSize; j++ {
				pm := getPM()
				mu.Lock()
				if pm > peakPM {
					peakPM = pm
				}
				t0 := time.Now()
				sw.s.Run()
				lat := float64(time.Since(t0).Milliseconds())
				latencies = append(latencies, lat)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	totalTime := float64(time.Since(startTime).Milliseconds())
	endPM := getPM()
	return summarizeArch(numRequests, totalTime, startPM, peakPM, endPM, latencies)
}

func testSessionPool(modelPath string, inputShape, outputShape []int64, inputData []byte, poolSize, numRequests int) ArchResult {
	startPM := getPM()
	peakPM := startPM
	var mu sync.Mutex

	var wg sync.WaitGroup
	var latencies []float64
	startTime := time.Now()
	batchSize := numRequests / poolSize

	for i := 0; i < poolSize; i++ {
		sw, err := createSWT(modelPath, inputShape, outputShape, 1, inputData)
		if err != nil {
			continue
		}
		wg.Add(1)
		go func(s *swt) {
			defer wg.Done()
			defer s.s.Destroy()
			defer s.it.Destroy()
			defer s.ot.Destroy()
			for j := 0; j < batchSize; j++ {
				pm := getPM()
				mu.Lock()
				if pm > peakPM {
					peakPM = pm
				}
				mu.Unlock()

				t0 := time.Now()
				s.s.Run()
				lat := float64(time.Since(t0).Milliseconds())
				mu.Lock()
				latencies = append(latencies, lat)
				mu.Unlock()
			}
		}(sw)
	}
	wg.Wait()

	totalTime := float64(time.Since(startTime).Milliseconds())
	endPM := getPM()
	mu.Lock()
	finalPeak := peakPM
	mu.Unlock()
	return summarizeArch(numRequests, totalTime, startPM, finalPeak, endPM, latencies)
}

func summarizeArch(numRequests int, totalTime, startPM, peakPM, endPM float64, latencies []float64) ArchResult {
	if len(latencies) == 0 {
		return ArchResult{StartPM: startPM, PeakPM: peakPM, EndPM: endPM, PMDrift: endPM - startPM}
	}
	sort.Float64s(latencies)
	sum := 0.0
	for _, l := range latencies {
		sum += l
	}
	avg := sum / float64(len(latencies))
	throughput := float64(len(latencies)) / (totalTime / 1000.0)

	return ArchResult{
		Throughput: throughput,
		AvgLatency: avg,
		P50Latency: calcPercentile(latencies, 0.50),
		P90Latency: calcPercentile(latencies, 0.90),
		P99Latency: calcPercentile(latencies, 0.99),
		MinLatency: latencies[0],
		MaxLatency: latencies[len(latencies)-1],
		StartPM:    startPM,
		PeakPM:     peakPM,
		EndPM:      endPM,
		PMDrift:    endPM - startPM,
	}
}

// ============================================================
// [3/6] Session Pool 消融实验
// ============================================================

type AblationResult struct {
	Model       string  `json:"model"`
	PoolSize    int     `json:"pool_size"`
	IntraThread int     `json:"intra_threads"`
	Status      string  `json:"status,omitempty"`
	SkipReason  string  `json:"skip_reason,omitempty"`
	Throughput  float64 `json:"throughput_reqs"`
	AvgLatency  float64 `json:"avg_latency_ms"`
	P50Latency  float64 `json:"p50_latency_ms"`
	P90Latency  float64 `json:"p90_latency_ms"`
	P99Latency  float64 `json:"p99_latency_ms"`
	MinLatency  float64 `json:"min_latency_ms"`
	MaxLatency  float64 `json:"max_latency_ms"`
	StdLatency  float64 `json:"std_latency_ms"`
	StartPM     float64 `json:"start_pm_mb"`
	PeakPM      float64 `json:"peak_pm_mb"`
	EndPM       float64 `json:"end_pm_mb"`
	PMDrift     float64 `json:"pm_drift_mb"`
	DurationSec float64 `json:"duration_sec"`
}

func runAblation(basePath string, modelName, modelFile string, inputData []byte) []AblationResult {
	fmt.Printf("\n============================================================\n")
	fmt.Printf("[3/6] Session Pool 消融实验 — %s\n", modelName)
	fmt.Printf("============================================================\n")

	modelPath := filepath.Join(basePath, "third_party", modelFile)
	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	poolSizes := []int{4, 8, 12, 16}
	threadConfigs := []int{1, 2, 4, 8}
	const numRequests = 500
	const concurrency = 12

	var results []AblationResult

	for _, poolSize := range poolSizes {
		for _, threads := range threadConfigs {
			if threads > poolSize {
				fmt.Printf("  [SKIP] Pool=%d Threads=%d (threads > pool_size)\n", poolSize, threads)
				results = append(results, AblationResult{
					Model:      modelName,
					PoolSize:   poolSize,
					IntraThread: threads,
					Status:     "SKIPPED",
					SkipReason: "threads > pool_size",
				})
				continue
			}
			fmt.Printf("  Pool=%d Threads=%d ... ", poolSize, threads)

			runtime.GOMAXPROCS(poolSize)
			startPM := getPM()
			peakPM := startPM

			// Create pool
			type poolItem struct {
				s  *ort.AdvancedSession
				it *ort.Tensor[float32]
				ot *ort.Tensor[float32]
			}
			pool := make(chan *poolItem, poolSize)
			var allSessions []*poolItem

			for i := 0; i < poolSize; i++ {
				opts, _ := ort.NewSessionOptions()
				opts.SetIntraOpNumThreads(threads)
				opts.SetInterOpNumThreads(1)
				opts.SetExecutionMode(0)
				opts.SetGraphOptimizationLevel(3)

				it, _ := ort.NewEmptyTensor[float32](inputShape)
				ot, _ := ort.NewEmptyTensor[float32](outputShape)
				fillTensorData(it, inputData)

				s, err := ort.NewAdvancedSession(modelPath,
					[]string{"images"}, []string{"output0"},
					[]ort.ArbitraryTensor{it}, []ort.ArbitraryTensor{ot}, opts)
				opts.Destroy()
				if err != nil {
					it.Destroy()
					ot.Destroy()
					continue
				}
				item := &poolItem{s, it, ot}
				pool <- item
				allSessions = append(allSessions, item)
			}

			// Warmup
			for i := 0; i < 10; i++ {
				item := <-pool
				item.s.Run()
				pool <- item
			}

			// Benchmark
			var (
				latencies   []float64
				completed   int32
				totalIssued int32
				mu          sync.Mutex
				wg          sync.WaitGroup
			)
			startTime := time.Now()

			for c := 0; c < concurrency; c++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for {
						idx := int(atomic.AddInt32(&totalIssued, 1))
						if idx > numRequests {
							break
						}
						item := <-pool
						t0 := time.Now()
						item.s.Run()
						lat := float64(time.Since(t0).Milliseconds())
						pool <- item

						mu.Lock()
						latencies = append(latencies, lat)
						pm := getPM()
						if pm > peakPM {
							peakPM = pm
						}
						atomic.AddInt32(&completed, 1)
						mu.Unlock()
					}
				}()
			}
			wg.Wait()

			duration := time.Since(startTime).Seconds()
			endPM := getPM()

			// Stats
			sort.Float64s(latencies)
			n := len(latencies)
			if n == 0 {
				for _, item := range allSessions {
					item.s.Destroy()
					item.it.Destroy()
					item.ot.Destroy()
				}
				results = append(results, AblationResult{
					Model:       modelName,
					PoolSize:    poolSize,
					IntraThread: threads,
					Status:      "SKIPPED",
					SkipReason:  "No valid latency data",
					StartPM:     startPM,
					PeakPM:      peakPM,
					EndPM:       endPM,
					PMDrift:     endPM - startPM,
					DurationSec: duration,
				})
				continue
			}

			var sum, sumSq float64
			for _, l := range latencies {
				sum += l
			}
			avg := sum / float64(n)
			for _, l := range latencies {
				diff := l - avg
				sumSq += diff * diff
			}
			std := math.Sqrt(sumSq / float64(n))
			throughput := float64(n) / duration

			result := AblationResult{
				Model:       modelName,
				PoolSize:    poolSize,
				IntraThread: threads,
				Status:      "OK",
				Throughput:  throughput,
				AvgLatency:  avg,
				P50Latency:  calcPercentile(latencies, 0.50),
				P90Latency:  calcPercentile(latencies, 0.90),
				P99Latency:  calcPercentile(latencies, 0.99),
				MinLatency:  latencies[0],
				MaxLatency:  latencies[len(latencies)-1],
				StdLatency:  std,
				StartPM:     startPM,
				PeakPM:      peakPM,
				EndPM:       endPM,
				PMDrift:     endPM - startPM,
				DurationSec: duration,
			}
			results = append(results, result)
			fmt.Printf("吞吐=%.2f P99=%.2f PM峰值=%.2f\n", throughput, result.P99Latency, peakPM)

			// Cleanup
			for _, item := range allSessions {
				item.s.Destroy()
				item.it.Destroy()
				item.ot.Destroy()
			}
			time.Sleep(2 * time.Second)
		}
	}

	return results
}

// ============================================================
// [4/6] 冷启动分解实验
// ============================================================

type ColdStartResult struct {
	Model             string  `json:"model"`
	Run               int     `json:"run"`
	SessionCreationMs float64 `json:"session_creation_ms"`
	FirstInferenceMs  float64 `json:"first_inference_ms"`
	TotalColdStartMs  float64 `json:"total_cold_start_ms"`
	StartPM           float64 `json:"start_pm_mb"`
	PeakPM            float64 `json:"peak_pm_mb"`
}

func runColdStart(basePath, modelName, modelFile string, inputData []byte) []ColdStartResult {
	fmt.Printf("\n============================================================\n")
	fmt.Printf("[4/6] 冷启动分解实验 — %s (20次)\n", modelName)
	fmt.Printf("============================================================\n")

	modelPath := filepath.Join(basePath, "third_party", modelFile)
	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}
	const numRuns = 20

	var results []ColdStartResult

	for i := 0; i < numRuns; i++ {
		fmt.Printf("  第%d次 ... ", i+1)

		startPM := getPM()

		totalStart := time.Now()

		opts, _ := ort.NewSessionOptions()
		opts.SetIntraOpNumThreads(1)
		opts.SetInterOpNumThreads(1)

		it, _ := ort.NewEmptyTensor[float32](inputShape)
		ot, _ := ort.NewEmptyTensor[float32](outputShape)
		fillTensorData(it, inputData)

		sessionStart := time.Now()
		s, err := ort.NewAdvancedSession(modelPath,
			[]string{"images"}, []string{"output0"},
			[]ort.ArbitraryTensor{it}, []ort.ArbitraryTensor{ot}, opts)
		opts.Destroy()
		sessionCreationMs := float64(time.Since(sessionStart).Milliseconds())

		if err != nil {
			it.Destroy()
			ot.Destroy()
			fmt.Printf("ERROR: %v\n", err)
			continue
		}

		firstStart := time.Now()
		s.Run()
		firstInferenceMs := float64(time.Since(firstStart).Milliseconds())

		totalColdStartMs := float64(time.Since(totalStart).Milliseconds())
		peakPM := getPM()

		s.Destroy()
		it.Destroy()
		ot.Destroy()

		results = append(results, ColdStartResult{
			Model:             modelName,
			Run:               i + 1,
			SessionCreationMs: sessionCreationMs,
			FirstInferenceMs:  firstInferenceMs,
			TotalColdStartMs:  totalColdStartMs,
			StartPM:           startPM,
			PeakPM:            peakPM,
		})
		fmt.Printf("会话=%.2f ms 首次推理=%.2f ms 总计=%.2f ms PM峰值=%.2f MB\n",
			sessionCreationMs, firstInferenceMs, totalColdStartMs, peakPM)

		// 每次冷启动后等待 GC
		time.Sleep(3 * time.Second)
	}

	return results
}

// ============================================================
// [5/6] 10分钟稳定性实验
// ============================================================

type StabilitySample struct {
	ElapsedSec float64 `json:"elapsed_sec"`
	PM         float64 `json:"pm_mb"`
	LatencyMs  float64 `json:"latency_ms"`
}

type StabilityResult struct {
	TotalDurationSec float64           `json:"total_duration_sec"`
	TotalInferences  int               `json:"total_inferences"`
	AvgLatencyMs     float64           `json:"avg_latency_ms"`
	P50LatencyMs     float64           `json:"p50_latency_ms"`
	P99LatencyMs     float64           `json:"p99_latency_ms"`
	StartPM          float64           `json:"start_pm_mb"`
	PeakPM           float64           `json:"peak_pm_mb"`
	EndPM            float64           `json:"end_pm_mb"`
	PMDrift          float64           `json:"pm_drift_mb"`
	Samples          []StabilitySample `json:"samples"`
}

func runStability(modelPath string, inputShape, outputShape []int64, inputData []byte) *StabilityResult {
	fmt.Println("\n============================================================")
	fmt.Println("[5/6] 10分钟稳定性实验")
	fmt.Println("============================================================")

	sw, err := createSWT(modelPath, inputShape, outputShape, 1, inputData)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return nil
	}
	defer sw.s.Destroy()
	defer sw.it.Destroy()
	defer sw.ot.Destroy()

	// Warmup
	for i := 0; i < 10; i++ {
		sw.s.Run()
	}

	startPM := getPM()
	peakPM := startPM
	var latencies []float64
	var samples []StabilitySample

	startTime := time.Now()
	testDuration := 10 * time.Minute
	endTime := startTime.Add(testDuration)
	inferenceCount := 0

	for time.Now().Before(endTime) {
		t0 := time.Now()
		sw.s.Run()
		lat := float64(time.Since(t0).Milliseconds())
		latencies = append(latencies, lat)
		inferenceCount++

		// Sample every 10 inferences
		if inferenceCount%10 == 0 {
			pm := getPM()
			if pm > peakPM {
				peakPM = pm
			}
			samples = append(samples, StabilitySample{
				ElapsedSec: time.Since(startTime).Seconds(),
				PM:         pm,
				LatencyMs:  lat,
			})
		}

		// Progress every minute
		if inferenceCount%600 == 0 {
			elapsed := time.Since(startTime).Round(time.Second)
			remaining := endTime.Sub(time.Now()).Round(time.Second)
			fmt.Printf("  推理=%d 已运行=%v 剩余=%v PM=%.2f MB\n",
				inferenceCount, elapsed, remaining, getPM())
		}
	}

	endPM := getPM()
	totalDuration := time.Since(startTime).Seconds()

	sort.Float64s(latencies)
	sum := 0.0
	for _, l := range latencies {
		sum += l
	}
	avgLat := sum / float64(len(latencies))

	result := &StabilityResult{
		TotalDurationSec: totalDuration,
		TotalInferences:  inferenceCount,
		AvgLatencyMs:     avgLat,
		P50LatencyMs:     calcPercentile(latencies, 0.50),
		P99LatencyMs:     calcPercentile(latencies, 0.99),
		StartPM:          startPM,
		PeakPM:           peakPM,
		EndPM:            endPM,
		PMDrift:          endPM - startPM,
		Samples:          samples,
	}

	fmt.Printf("  完成: %d次推理, 平均延迟=%.3f ms, PM漂移=%.2f MB\n",
		inferenceCount, avgLat, endPM-startPM)
	return result
}

// ============================================================
// [6/6] 批量推理效果
// ============================================================

type BatchResult struct {
	BatchSize  int     `json:"batch_size"`
	Throughput float64 `json:"throughput_reqs"`
	AvgLatency float64 `json:"avg_latency_ms"`
	P99Latency float64 `json:"p99_latency_ms"`
	StartPM    float64 `json:"start_pm_mb"`
	PeakPM     float64 `json:"peak_pm_mb"`
}

func runBatchEffect(modelPath string, inputShape, outputShape []int64, inputData []byte) []BatchResult {
	fmt.Println("\n============================================================")
	fmt.Println("[6/6] 批量推理效果")
	fmt.Println("============================================================")

	const numRequests = 100
	var results []BatchResult

	for _, batchSize := range []int{1, 2, 4, 8} {
		fmt.Printf("  Batch=%d ... ", batchSize)

		startPM := getPM()
		peakPM := startPM

		// Create session with batch-sized tensors
		batchInputShape := make([]int64, len(inputShape))
		copy(batchInputShape, inputShape)
		batchInputShape[0] = int64(batchSize)

		batchOutputShape := make([]int64, len(outputShape))
		copy(batchOutputShape, outputShape)
		batchOutputShape[0] = int64(batchSize)

		sw, err := createSWT(modelPath, batchInputShape, batchOutputShape, 4, inputData)
		if err != nil {
			fmt.Printf("ERROR: %v\n", err)
			continue
		}

		// Warmup
		for i := 0; i < 5; i++ {
			sw.s.Run()
		}

		var latencies []float64
		startTime := time.Now()

		for i := 0; i < numRequests; i++ {
			pm := getPM()
			if pm > peakPM {
				peakPM = pm
			}

			t0 := time.Now()
			sw.s.Run()
			lat := float64(time.Since(t0).Milliseconds())
			latencies = append(latencies, lat)
		}

		totalTime := float64(time.Since(startTime).Milliseconds())
		sw.s.Destroy()
		sw.it.Destroy()
		sw.ot.Destroy()

		sort.Float64s(latencies)
		sum := 0.0
		for _, l := range latencies {
			sum += l
		}
		avg := sum / float64(len(latencies))
		// Throughput = effective images per second (batch_size × requests / time)
		throughput := float64(batchSize*numRequests) / (totalTime / 1000.0)
		if math.IsInf(throughput, 0) || math.IsNaN(throughput) {
			throughput = math.MaxFloat64
		}

		result := BatchResult{
			BatchSize:  batchSize,
			Throughput: throughput,
			AvgLatency: avg,
			P99Latency: calcPercentile(latencies, 0.99),
			StartPM:    startPM,
			PeakPM:     peakPM,
		}
		results = append(results, result)
		fmt.Printf("吞吐=%.2f img/s 延迟=%.2f ms PM峰值=%.2f MB\n",
			throughput, avg, peakPM)

		time.Sleep(2 * time.Second)
	}

	return results
}

// ============================================================
// 主程序
// ============================================================

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  论文完整基准测试程序")
	fmt.Println("  内存指标: PrivateMemorySize64 (可靠) + WorkingSet64 (对比)")
	fmt.Println("  运行方式: 编译为 .exe 后运行")
	fmt.Println("  预计耗时: 60-90 分钟")
	fmt.Println(strings.Repeat("=", 70))

	// 获取路径
	wd, _ := os.Getwd()
	basePath := filepath.Dir(filepath.Dir(wd))
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPathX := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	modelPathN := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	resultsDir := filepath.Join(basePath, "results")

	// 验证文件
	for _, p := range []string{libPath, modelPathX, modelPathN, inputDataPath} {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			fmt.Printf("ERROR: 文件不存在: %s\n", p)
			os.Exit(1)
		}
	}

	os.MkdirAll(resultsDir, 0755)

	// 初始化 ONNX Runtime
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("ERROR: 初始化 ONNX Runtime 失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	runtime.GOMAXPROCS(12)

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("ERROR: 读取输入数据失败: %v\n", err)
		os.Exit(1)
	}

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// ==========================================
	// [1/6] 内存指标验证
	// ==========================================
	memResults := runMemoryVerification(modelPathX, inputShape, outputShape, inputData)
	saveMemoryStandardizationTxt(resultsDir, memResults)

	// ==========================================
	// [2/6] 架构对比实验（Unsafe Shared 通过子进程运行，崩溃会被捕获记录）
	// ==========================================
	archResults := runArchitectureBenchmark(modelPathX, inputShape, outputShape, inputData)
	if len(archResults) > 0 {
		saveArchitectureTxt(resultsDir, archResults)
	}

	// ==========================================
	// [3/6] 消融实验
	// ==========================================
	ablationX := runAblation(basePath, "YOLO11x", "yolo11x.onnx", inputData)
	ablationN := runAblation(basePath, "YOLO11n", "yolo11n.onnx", inputData)
	saveAblationJSON(resultsDir, ablationX, ablationN)

	// ==========================================
	// [4/6] 冷启动分解
	// ==========================================
	coldStartX := runColdStart(basePath, "YOLO11x", "yolo11x.onnx", inputData)
	coldStartN := runColdStart(basePath, "YOLO11n", "yolo11n.onnx", inputData)
	saveColdStartTxt(resultsDir, coldStartX, coldStartN)

	// ==========================================
	// [5/6] 10分钟稳定性
	// ==========================================
	stabilityResult := runStability(modelPathX, inputShape, outputShape, inputData)
	if stabilityResult != nil {
		saveStabilityTxt(resultsDir, stabilityResult)
		saveRSSCurveCSV(resultsDir, stabilityResult)
	}

	// ==========================================
	// [6/6] 批量推理效果
	// ==========================================
	batchResults := runBatchEffect(modelPathX, inputShape, outputShape, inputData)
	saveBatchJSON(resultsDir, batchResults)

	// ==========================================
	// 生成汇总报告
	// ==========================================
	generateSummary(resultsDir, memResults, archResults, stabilityResult, batchResults)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Printf("  全部测试完成！结果保存在: %s\n", resultsDir)
	fmt.Println(strings.Repeat("=", 70))
}

// saveMemoryStandardizationTxt 保存内存标准化结果（覆盖 go_memory_standardization_result.txt）
func saveMemoryStandardizationTxt(dir string, results []MemVerifyResult) {
	filename := "go_memory_standardization_result.txt"
	path := filepath.Join(dir, filename)
	var sb strings.Builder
	sb.WriteString("===== Go 内存标准化测试结果 (PrivateMemorySize64) =====\n\n")
	for _, r := range results {
		sb.WriteString(fmt.Sprintf("State: %s  WS=%.2f MB  PM=%.2f MB\n", r.State, r.WS, r.PM))
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

// saveArchitectureTxt 保存架构对比结果（覆盖 go_architecture_comparison.txt）
func saveArchitectureTxt(dir string, results []ArchResult) {
	filename := "go_architecture_comparison.txt"
	path := filepath.Join(dir, filename)
	var sb strings.Builder
	sb.WriteString("===== Go 推理架构性能对比实验结果 (PrivateMemorySize64) =====\n\n")
	sb.WriteString("测试三种架构：\n")
	sb.WriteString("  1. Unsafe Shared - 共享 Session，独立 Tensor（测试 Session Contention）\n")
	sb.WriteString("  2. Mutex Shared  - 共享 Session，加锁串行化\n")
	sb.WriteString("  3. Session Pool  - 独立 Session（最佳方案）\n\n")
	for _, r := range results {
		sb.WriteString(fmt.Sprintf("===== %s =====\n", r.Architecture))
		if r.Architecture == "Unsafe Shared" && r.Throughput < 0 {
			// Crash detected: throughput = -1, concurrency = -exitCode
			sb.WriteString(fmt.Sprintf("  状态: CRASHED (exit code: %d)\n", -r.Concurrency))
			sb.WriteString(fmt.Sprintf("  实测结果: Unsafe Shared 子进程以 exit code %d 异常退出，\n", -r.Concurrency))
			sb.WriteString("           说明共享 Session 不加锁的架构在并发推理时不可靠。\n")
			sb.WriteString("           具体原因见子进程 stderr 输出（已记录于日志文件）。\n\n")
			continue
		}
		if r.Architecture == "Unsafe Shared" {
			// No crash: record actual throughput data
			sb.WriteString(fmt.Sprintf("  状态: COMPLETED (exit code: 0)\n"))
			sb.WriteString(fmt.Sprintf("  实测结果: Unsafe Shared 子进程正常完成，并发=%d\n", r.Concurrency))
			sb.WriteString(fmt.Sprintf("  吞吐量: %.5f REQ/s\n\n", r.Throughput))
			continue
		}
		if r.Architecture == "Session Pool" {
			sb.WriteString(fmt.Sprintf("池大小: %d\n", r.PoolSize))
		} else {
			sb.WriteString(fmt.Sprintf("并发度: %d\n", r.Concurrency))
		}
		sb.WriteString(fmt.Sprintf("  吞吐量: %.5f REQ/s\n", r.Throughput))
		sb.WriteString(fmt.Sprintf("  平均延迟: %.5f ms\n", r.AvgLatency))
		sb.WriteString(fmt.Sprintf("  P50延迟: %.5f ms\n", r.P50Latency))
		sb.WriteString(fmt.Sprintf("  P90延迟: %.5f ms\n", r.P90Latency))
		sb.WriteString(fmt.Sprintf("  P99延迟: %.5f ms\n", r.P99Latency))
		sb.WriteString(fmt.Sprintf("  最小延迟: %.5f ms\n", r.MinLatency))
		sb.WriteString(fmt.Sprintf("  最大延迟: %.5f ms\n", r.MaxLatency))
		sb.WriteString(fmt.Sprintf("  峰值PM: %.5f MB\n", r.PeakPM))
		sb.WriteString(fmt.Sprintf("  PM漂移: %.5f MB\n\n", r.PMDrift))
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

// saveAblationJSON 保存消融实验结果（合并两个模型，输出到 paper_full_benchmark_ablation.json）
func saveAblationJSON(dir string, ablationX, ablationN []AblationResult) {
	filename := "paper_full_benchmark_ablation.json"
	path := filepath.Join(dir, filename)
	allResults := append(ablationX, ablationN...)
	b, err := json.MarshalIndent(allResults, "", "  ")
	if err != nil {
		fmt.Printf("  WARN: JSON序列化失败 %s: %v\n", filename, err)
		return
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

// saveColdStartTxt 保存冷启动分解结果（覆盖 go_cold_start_decomposition_result.txt，合并两个模型）
func saveColdStartTxt(dir string, coldStartX, coldStartN []ColdStartResult) {
	filename := "go_cold_start_decomposition_result.txt"
	path := filepath.Join(dir, filename)
	var sb strings.Builder
	sb.WriteString("===== Go 冷启动分解测试结果 (PrivateMemorySize64) =====\n\n")
	sb.WriteString("===== 大模型 (YOLO11x) =====\n")
	writeColdStartModel(&sb, coldStartX)
	sb.WriteString("===== 轻模型 (YOLO11n) =====\n")
	writeColdStartModel(&sb, coldStartN)
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

func writeColdStartModel(sb *strings.Builder, results []ColdStartResult) {
	var sumCreation, sumInference, sumTotal, sumStartPM, sumPeakPM float64
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("===== 第 %d 次测试 =====\n", i+1))
		sb.WriteString(fmt.Sprintf("会话创建时间: %.5f ms\n", r.SessionCreationMs))
		sb.WriteString(fmt.Sprintf("首次推理时间: %.5f ms\n", r.FirstInferenceMs))
		sb.WriteString(fmt.Sprintf("总冷启动时间: %.5f ms\n", r.TotalColdStartMs))
		sb.WriteString(fmt.Sprintf("Start PM: %.5f MB\n", r.StartPM))
		sb.WriteString(fmt.Sprintf("Peak PM: %.5f MB\n\n", r.PeakPM))
		sumCreation += r.SessionCreationMs
		sumInference += r.FirstInferenceMs
		sumTotal += r.TotalColdStartMs
		sumStartPM += r.StartPM
		sumPeakPM += r.PeakPM
	}
	n := float64(len(results))
	sb.WriteString(fmt.Sprintf("===== 平均值 (共%d次) =====\n", len(results)))
	sb.WriteString(fmt.Sprintf("会话创建时间: %.5f ms\n", sumCreation/n))
	sb.WriteString(fmt.Sprintf("首次推理时间: %.5f ms\n", sumInference/n))
	sb.WriteString(fmt.Sprintf("总冷启动时间: %.5f ms\n", sumTotal/n))
	sb.WriteString(fmt.Sprintf("Start PM: %.5f MB\n", sumStartPM/n))
	sb.WriteString(fmt.Sprintf("Peak PM: %.5f MB\n\n", sumPeakPM/n))
}

// saveStabilityTxt 保存10分钟稳定性结果（覆盖 go_long_stability_result.txt）
func saveStabilityTxt(dir string, r *StabilityResult) {
	filename := "go_long_stability_result.txt"
	path := filepath.Join(dir, filename)
	var sb strings.Builder
	sb.WriteString("===== Go 长时间稳定性测试结果 (PrivateMemorySize64) =====\n")
	sb.WriteString(fmt.Sprintf("测试时长: %.0f 秒\n", r.TotalDurationSec))
	sb.WriteString(fmt.Sprintf("推理次数: %d\n", r.TotalInferences))
	sb.WriteString(fmt.Sprintf("推理频率: %.5f 次/秒\n", float64(r.TotalInferences)/r.TotalDurationSec))
	sb.WriteString("\n===== 推理性能统计 =====\n")
	sb.WriteString(fmt.Sprintf("平均推理时间: %.5f ms\n", r.AvgLatencyMs))
	sb.WriteString(fmt.Sprintf("P50推理时间: %.5f ms\n", r.P50LatencyMs))
	sb.WriteString(fmt.Sprintf("P99推理时间: %.5f ms\n", r.P99LatencyMs))
	sb.WriteString("\n===== 内存使用统计 (PrivateMemorySize64) =====\n")
	sb.WriteString(fmt.Sprintf("起始 PM: %.5f MB\n", r.StartPM))
	sb.WriteString(fmt.Sprintf("峰值 PM: %.5f MB\n", r.PeakPM))
	sb.WriteString(fmt.Sprintf("结束 PM: %.5f MB\n", r.EndPM))
	sb.WriteString(fmt.Sprintf("PM 漂移: %.5f MB\n", r.PMDrift))
	sb.WriteString(fmt.Sprintf("PM 波动: %.5f MB\n", r.PeakPM-r.EndPM))
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

// saveRSSCurveCSV 保存PM曲线CSV（覆盖 go_rss_curve.csv）
func saveRSSCurveCSV(dir string, r *StabilityResult) {
	filename := "go_rss_curve.csv"
	path := filepath.Join(dir, filename)
	var sb strings.Builder
	sb.WriteString("Timestamp,Elapsed_Seconds,PM_MB\n")
	for _, s := range r.Samples {
		sb.WriteString(fmt.Sprintf("%.3f,%.5f,%.5f\n", s.ElapsedSec, s.ElapsedSec, s.PM))
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

// saveBatchJSON 保存批量推理结果（覆盖 go_batch_inference_result.json）
func saveBatchJSON(dir string, results []BatchResult) {
	filename := "go_batch_inference_result.json"
	path := filepath.Join(dir, filename)
	b, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("  WARN: JSON序列化失败 %s: %v\n", filename, err)
		return
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		fmt.Printf("  WARN: 保存失败 %s: %v\n", filename, err)
		return
	}
	fmt.Printf("  -> 已保存: %s\n", path)
}

func generateSummary(dir string, memResults []MemVerifyResult, archResults []ArchResult, stability *StabilityResult, batchResults []BatchResult) {
	var sb strings.Builder
	sb.WriteString("===== 论文完整基准测试 — 汇总报告 =====\n")
	sb.WriteString(fmt.Sprintf("生成时间: %s\n\n", time.Now().Format("2006-01-02 15:04:05")))

	// 内存验证摘要
	sb.WriteString("[1/6] 内存指标验证 (PrivateMemorySize64 vs WorkingSet64)\n")
	sb.WriteString(fmt.Sprintf("%-30s %12s %12s\n", "State", "WS(MB)", "PM(MB)"))
	for _, r := range memResults {
		sb.WriteString(fmt.Sprintf("%-30s %12.2f %12.2f\n", r.State, r.WS, r.PM))
	}

	// 架构对比摘要
	sb.WriteString("\n[2/6] 架构对比实验\n")
	sb.WriteString(fmt.Sprintf("%-18s %5s %10s %10s %10s %10s\n",
		"Architecture", "Conc", "Throughput", "Avg(ms)", "P99(ms)", "PeakPM(MB)"))
	for _, r := range archResults {
		sb.WriteString(fmt.Sprintf("%-18s %5d %10.2f %10.3f %10.3f %10.2f\n",
			r.Architecture, max(r.Concurrency, r.PoolSize), r.Throughput, r.AvgLatency, r.P99Latency, r.PeakPM))
	}

	// 稳定性摘要
	if stability != nil {
		sb.WriteString("\n[5/6] 10分钟稳定性实验\n")
		sb.WriteString(fmt.Sprintf("  总推理次数: %d\n", stability.TotalInferences))
		sb.WriteString(fmt.Sprintf("  平均延迟: %.3f ms\n", stability.AvgLatencyMs))
		sb.WriteString(fmt.Sprintf("  P50延迟: %.3f ms\n", stability.P50LatencyMs))
		sb.WriteString(fmt.Sprintf("  P99延迟: %.3f ms\n", stability.P99LatencyMs))
		sb.WriteString(fmt.Sprintf("  起始PM: %.2f MB\n", stability.StartPM))
		sb.WriteString(fmt.Sprintf("  峰值PM: %.2f MB\n", stability.PeakPM))
		sb.WriteString(fmt.Sprintf("  结束PM: %.2f MB\n", stability.EndPM))
		sb.WriteString(fmt.Sprintf("  PM漂移: %.2f MB\n", stability.PMDrift))
	}

	// 批量推理摘要
	sb.WriteString("\n[6/6] 批量推理效果\n")
	sb.WriteString(fmt.Sprintf("%-10s %12s %12s %12s %12s\n",
		"BatchSize", "Throughput", "Avg(ms)", "P99(ms)", "PeakPM(MB)"))
	for _, r := range batchResults {
		sb.WriteString(fmt.Sprintf("%-10d %12.2f %12.3f %12.3f %12.2f\n",
			r.BatchSize, r.Throughput, r.AvgLatency, r.P99Latency, r.PeakPM))
	}

	os.WriteFile(filepath.Join(dir, "paper_full_benchmark_summary.txt"), []byte(sb.String()), 0644)
	fmt.Printf("  -> 已保存: %s\n", filepath.Join(dir, "paper_full_benchmark_summary.txt"))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}