package main

import (
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

type poolWorker struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

type HourlySnapshot struct {
	Hour         int     `json:"hour"`
	TotalInfer   int     `json:"total_infer"`
	HourlyInfer  int     `json:"hourly_infer"`
	Throughput   float64 `json:"throughput"`
	AvgLatencyMs float64 `json:"avg_latency_ms"`
	P50LatencyMs float64 `json:"p50_latency_ms"`
	P99LatencyMs float64 `json:"p99_latency_ms"`
	PM_MB        float64 `json:"pm_mb"`
	GCNum        uint32  `json:"gc_num"`
	Timestamp    string  `json:"timestamp"`
}

type FinalResult struct {
	TestName           string           `json:"test_name"`
	Model              string           `json:"model"`
	DurationHours      float64          `json:"duration_hours"`
	TotalInferences    int              `json:"total_inferences"`
	AvgLatencyMs       float64          `json:"avg_latency_ms"`
	P50LatencyMs       float64          `json:"p50_latency_ms"`
	P99LatencyMs       float64          `json:"p99_latency_ms"`
	MinLatencyMs       float64          `json:"min_latency_ms"`
	MaxLatencyMs       float64          `json:"max_latency_ms"`
	StartPM_MB         float64          `json:"start_pm_mb"`
	EndPM_MB           float64          `json:"end_pm_mb"`
	PeakPM_MB          float64          `json:"peak_pm_mb"`
	MinPM_MB           float64          `json:"min_pm_mb"`
	PMDriftMB          float64          `json:"pm_drift_mb"`
	DriftRateMBPerHour float64          `json:"drift_rate_mb_per_hour"`
	TotalGCNum         uint32           `json:"total_gc_num"`
	PerfDegradePct     float64          `json:"perf_degrade_pct"`
	HourlySnapshots    []HourlySnapshot `json:"hourly_snapshots"`
	StartTime          string           `json:"start_time"`
	EndTime            string           `json:"end_time"`
}

func createWorker(modelPath string, inputShape, outputShape []int64, inputData []byte, intraOp int) (*poolWorker, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	opts.SetIntraOpNumThreads(intraOp)
	opts.SetInterOpNumThreads(1)
	opts.SetExecutionMode(0)

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入Tensor失败: %w", err)
	}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出Tensor失败: %w", err)
	}

	// fill input data
	fillInputData(inputTensor, inputData)

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		opts,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("创建Session失败: %w", err)
	}

	return &poolWorker{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

func fillInputData(tensor *ort.Tensor[float32], inputData []byte) {
	floatData := tensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

func getPM() float64 {
	return memutil.PrivateMemoryMB()
}

func getGCStats() uint32 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.NumGC
}

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

func main() {
	// 支持命令行参数：go_72h_stability.exe [小时数]
	totalHours := 72
	if len(os.Args) > 1 {
		if h, err := strconv.Atoi(os.Args[1]); err == nil && h > 0 {
			totalHours = h
		}
	}

	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("  Go %dh Long-Term Stability Test\n", totalHours)
	fmt.Println("  Memory metric: PrivateMemorySize64")
	fmt.Println("  Architecture: Time-based loop, Single Session (poolSize=1, intraOp=12, same caliber as Python)")
	fmt.Println(strings.Repeat("=", 70))

	wd, _ := os.Getwd()
	basePath := filepath.Dir(filepath.Dir(wd))
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")
	resultsDir := filepath.Join(basePath, "results")

	for _, p := range []string{libPath, modelPath, inputDataPath} {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			fmt.Printf("ERROR: file not found: %s\n", p)
			os.Exit(1)
		}
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("ERROR: init failed: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	runtime.GOMAXPROCS(12)

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("ERROR: read input data failed: %v\n", err)
		os.Exit(1)
	}

	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// Session Pool
	fmt.Println("\nCreating Single Session (poolSize=1, intraOp=12)...")
	const poolSize = 1
	pool := make(chan *poolWorker, poolSize)
	var allWorkers []*poolWorker

	for i := 0; i < poolSize; i++ {
		w, err := createWorker(modelPath, inputShape, outputShape, inputData, 12)
		if err != nil {
			fmt.Printf("ERROR: Worker %d creation failed: %v\n", i, err)
			os.Exit(1)
		}
		pool <- w
		allWorkers = append(allWorkers, w)
		fmt.Printf("  Worker %d created successfully\n", i+1)
	}

	// Warmup
	fmt.Print("Warmup...")
	for i := 0; i < 20; i++ {
		w := <-pool
		fillInputData(w.inputTensor, inputData)
		w.session.Run()
		pool <- w
	}
	fmt.Println(" done")

	// Initial state
	startTime := time.Now()
	startPM := getPM()
	startGCNum := getGCStats()
	peakPM := startPM
	minPM := startPM
	endTime := startTime.Add(time.Duration(totalHours) * time.Hour)

	fmt.Printf("\nStart time:  %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("Start PM:    %.2f MB\n", startPM)
	fmt.Printf("End time:    %s\n", endTime.Format("2006-01-02 15:04:05"))
	fmt.Println(strings.Repeat("=", 70))

	// CSV
	csvPath := filepath.Join(resultsDir, fmt.Sprintf("go_stability_%dh_detailed.csv", totalHours))
	csvFile, err := os.Create(csvPath)
	if err != nil {
		fmt.Printf("ERROR: create CSV failed: %v\n", err)
		os.Exit(1)
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()
	csvWriter.Write([]string{
		"Hour", "TotalInfer", "HourlyInfer", "Throughput(sec)",
		"AvgLatency(ms)", "P50Latency(ms)", "P99Latency(ms)",
		"PM_MB", "GCNum", "Timestamp",
	})

	// ============================================================
	// Time-based main loop (THE FIX: truly run for N hours)
	// ============================================================
	var (
		allLatencies    []float64
		hourlySnapshots []HourlySnapshot
		pmMutex         sync.Mutex
		totalInferences int
	)

	const concurrency = 1

	for hour := 1; hour <= totalHours; hour++ {
		hourStart := time.Now()
		hourEnd := startTime.Add(time.Duration(hour) * time.Hour)
		if hourEnd.After(endTime) {
			hourEnd = endTime
		}

		// If the hour end has already passed, skip
		if !hourEnd.After(hourStart) {
			break
		}

		var hourlyLatencies []float64
		var latMutex sync.Mutex
		var completed int32
		var wg sync.WaitGroup

		// Spawn concurrent workers — each runs inferences continuously until hourEnd
		for c := 0; c < concurrency; c++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for time.Now().Before(hourEnd) {
					w := <-pool

					fillInputData(w.inputTensor, inputData)
					t0 := time.Now()
					w.session.Run()
					lat := float64(time.Since(t0).Nanoseconds()) / 1e6

					pool <- w

					latMutex.Lock()
					hourlyLatencies = append(hourlyLatencies, lat)
					latMutex.Unlock()

					atomic.AddInt32(&completed, 1)

					if atomic.LoadInt32(&completed)%100 == 0 {
						pm := getPM()
						pmMutex.Lock()
						if pm > peakPM {
							peakPM = pm
						}
						if pm < minPM {
							minPM = pm
						}
						pmMutex.Unlock()
					}
				}
			}()
		}
		wg.Wait()

		hourDuration := time.Since(hourStart).Seconds()
		if hourDuration < 1 {
			hourDuration = 1
		}

		hourlyInfer := len(hourlyLatencies)
		totalInferences += hourlyInfer

		// Hourly statistics
		sort.Float64s(hourlyLatencies)
		allLatencies = append(allLatencies, hourlyLatencies...)

		var hSum float64
		for _, l := range hourlyLatencies {
			hSum += l
		}
		hAvg := 0.0
		if hourlyInfer > 0 {
			hAvg = hSum / float64(hourlyInfer)
		}
		hThroughput := float64(hourlyInfer) / hourDuration

		currentPM := getPM()
		pmMutex.Lock()
		if currentPM > peakPM {
			peakPM = currentPM
		}
		if currentPM < minPM {
			minPM = currentPM
		}
		pmMutex.Unlock()

		gcNum := getGCStats() - startGCNum

		p50 := calcPercentile(hourlyLatencies, 0.50)
		p99 := calcPercentile(hourlyLatencies, 0.99)

		snapshot := HourlySnapshot{
			Hour:         hour,
			TotalInfer:   totalInferences,
			HourlyInfer:  hourlyInfer,
			Throughput:   hThroughput,
			AvgLatencyMs: hAvg,
			P50LatencyMs: p50,
			P99LatencyMs: p99,
			PM_MB:        currentPM,
			GCNum:        gcNum,
			Timestamp:    time.Now().Format("2006-01-02 15:04:05"),
		}
		hourlySnapshots = append(hourlySnapshots, snapshot)

		// CSV
		csvWriter.Write([]string{
			fmt.Sprintf("%d", hour),
			fmt.Sprintf("%d", totalInferences),
			fmt.Sprintf("%d", hourlyInfer),
			fmt.Sprintf("%.6f", hThroughput),
			fmt.Sprintf("%.3f", hAvg),
			fmt.Sprintf("%.3f", p50),
			fmt.Sprintf("%.3f", p99),
			fmt.Sprintf("%.2f", currentPM),
			fmt.Sprintf("%d", gcNum),
			snapshot.Timestamp,
		})
		csvWriter.Flush()

		// Console output
		elapsed := time.Since(startTime)
		remaining := endTime.Sub(time.Now())
		if remaining < 0 {
			remaining = 0
		}
		fmt.Printf("Hour %2d/%d | infer=%d(+%d) | %.1f/sec | p50=%.0fms p99=%.0fms | PM=%.0fMB | elapsed=%s | remaining=%s\n",
			hour, totalHours, totalInferences, hourlyInfer, hThroughput, p50, p99, currentPM,
			elapsed.Round(time.Minute), remaining.Round(time.Minute))
	}

	// ============================================================
	// Final statistics
	// ============================================================
	actualEndTime := time.Now()
	endPM := getPM()
	endGCNum := getGCStats() - startGCNum
	durationHours := actualEndTime.Sub(startTime).Hours()

	sort.Float64s(allLatencies)
	var sumLat float64
	for _, l := range allLatencies {
		sumLat += l
	}
	avgLat := 0.0
	if len(allLatencies) > 0 {
		avgLat = sumLat / float64(len(allLatencies))
	}

	// Performance degradation (first 10% vs last 10%)
	perfDegrade := 0.0
	sampleSize := len(allLatencies) / 10
	if sampleSize > 0 && len(allLatencies) >= 2*sampleSize {
		var earlySum, lateSum float64
		for i := 0; i < sampleSize; i++ {
			earlySum += allLatencies[i]
			lateSum += allLatencies[len(allLatencies)-1-i]
		}
		if earlySum > 0 {
			perfDegrade = ((lateSum/float64(sampleSize) - earlySum/float64(sampleSize)) / (earlySum / float64(sampleSize))) * 100
		}
	}

	driftRate := 0.0
	if durationHours > 0 {
		driftRate = (endPM - startPM) / durationHours
	}

	minLat := 0.0
	maxLat := 0.0
	if len(allLatencies) > 0 {
		minLat = allLatencies[0]
		maxLat = allLatencies[len(allLatencies)-1]
	}

	result := FinalResult{
		TestName:           fmt.Sprintf("%dh_Long_Stability", totalHours),
		Model:              "YOLO11x",
		DurationHours:      durationHours,
		TotalInferences:    totalInferences,
		AvgLatencyMs:       avgLat,
		P50LatencyMs:       calcPercentile(allLatencies, 0.50),
		P99LatencyMs:       calcPercentile(allLatencies, 0.99),
		MinLatencyMs:       minLat,
		MaxLatencyMs:       maxLat,
		StartPM_MB:         startPM,
		EndPM_MB:           endPM,
		PeakPM_MB:          peakPM,
		MinPM_MB:           minPM,
		PMDriftMB:          endPM - startPM,
		DriftRateMBPerHour: driftRate,
		TotalGCNum:         endGCNum,
		PerfDegradePct:     perfDegrade,
		HourlySnapshots:    hourlySnapshots,
		StartTime:          startTime.Format("2006-01-02 15:04:05"),
		EndTime:            actualEndTime.Format("2006-01-02 15:04:05"),
	}

	// Cleanup
	for _, w := range allWorkers {
		w.session.Destroy()
		w.inputTensor.Destroy()
		w.outputTensor.Destroy()
	}

	// Output results
	fmt.Printf("\n" + strings.Repeat("=", 70) + "\n")
	fmt.Printf("  %dh Stability Test — COMPLETE\n", totalHours)
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("  Total duration:   %.2f hours\n", durationHours)
	fmt.Printf("  Total inferences: %d\n", totalInferences)
	fmt.Printf("  Avg latency:      %.1f ms\n", avgLat)
	fmt.Printf("  P50 latency:      %.1f ms\n", calcPercentile(allLatencies, 0.50))
	fmt.Printf("  P99 latency:      %.1f ms\n", calcPercentile(allLatencies, 0.99))
	fmt.Printf("  Start PM:         %.2f MB\n", startPM)
	fmt.Printf("  End PM:           %.2f MB\n", endPM)
	fmt.Printf("  PM drift:         %.2f MB (%.4f MB/hour)\n", endPM-startPM, driftRate)
	fmt.Printf("  Peak PM:          %.2f MB\n", peakPM)
	fmt.Printf("  Min PM:           %.2f MB\n", minPM)
	fmt.Printf("  Performance Δ:    %.2f%%\n", perfDegrade)
	fmt.Printf("  GC count:         %d\n", endGCNum)

	// Save JSON
	jsonPath := filepath.Join(resultsDir, fmt.Sprintf("go_stability_%dh_result.json", totalHours))
	jsonData, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile(jsonPath, jsonData, 0644)
	fmt.Printf("\n  JSON result: %s\n", jsonPath)
	fmt.Printf("  CSV data:    %s\n", csvPath)
	fmt.Println("\nTest completed!")
}
