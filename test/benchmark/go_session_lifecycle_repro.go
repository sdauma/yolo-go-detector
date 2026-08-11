// go_session_lifecycle_repro.go
// Session 生命周期内存累积复现实验（Go 侧对照）
//
// 目的：受控复现 Session 生命周期内存漂移现象类（ORT issue #27089 相关）
// 的核心现象——“反复创建/销毁 Session 导致内存累积”——并验证其 runtime 依赖性：
//   - Python/glibc：反复建毁 → 分配器囤积 → 内存单调增长（见 python_session_lifecycle_repro.py）
//   - Go/ORT：销毁即归还（尤其本引擎默认关闭 Arena），故“临时会话”内存可控
//
// 设计：
//   - 自变量1：模式 {per-request(每轮建毁), pool(预建1个复用)}
//   - 自变量2：Arena {ON, OFF}
//   - 因变量：PM(PrivateMemorySize64) 随周期变化、峰值PM、漂移
//   - 控制：YOLO11x、intra_op=1、inter_op=1、GOMAXPROCS=6、每轮推理1次
//
// 预期（假设，供论文核对）：
//   - per-request + Arena OFF → 平（引擎默认，印证 §4.1 临时会话漂移≈0）
//   - per-request + Arena ON  → 可能小幅增长或平（ORT Arena 行为，数据为准）
//   - pool(两档)              → 平/有界
// Go 侧总体平，与 Python 侧增长形成对照，证“runtime 依赖性”。

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/test/benchmark/memutil"
)

func getProcessPM() float64 { return memutil.PrivateMemoryMB() }

type sessionWithTensors struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputData    []byte
}

func createSessionWithTensorsArena(
	modelPath string,
	inputData []byte,
	inputShape []int64,
	outputShape []int64,
	intraOpThreads int,
	arenaEnabled bool,
) (*sessionWithTensors, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建 SessionOptions 失败: %v", err)
	}
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(intraOpThreads)
	opts.SetInterOpNumThreads(1)
	if err := opts.SetMemPattern(false); err != nil {
		return nil, fmt.Errorf("SetMemPattern 失败: %v", err)
	}
	if err := opts.SetCpuMemArena(arenaEnabled); err != nil {
		return nil, fmt.Errorf("SetCpuMemArena 失败: %v", err)
	}
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("创建输入 Tensor 失败: %v", err)
	}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("创建输出 Tensor 失败: %v", err)
	}
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
		return nil, fmt.Errorf("创建 Session 失败: %v", err)
	}
	return &sessionWithTensors{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputData:    inputData,
	}, nil
}

func fillInputData(inputTensor *ort.Tensor[float32], inputData []byte) {
	floatData := inputTensor.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

func runInference(swt *sessionWithTensors) error {
	fillInputData(swt.inputTensor, swt.inputData)
	return swt.session.Run()
}

func destroySession(swt *sessionWithTensors) {
	if swt.session != nil {
		swt.session.Destroy()
	}
	if swt.inputTensor != nil {
		swt.inputTensor.Destroy()
	}
	if swt.outputTensor != nil {
		swt.outputTensor.Destroy()
	}
}

// testPerRequest: 反复 建→推→毁 模式
func testPerRequest(modelPath string, inputData []byte, inputShape, outputShape []int64,
	cycles, intraOp int, arenaEnabled bool) (series []float64, start, peak, end float64) {
	arenaStr := "ON"
	if !arenaEnabled {
		arenaStr = "OFF"
	}
	fmt.Printf("  [per-request arena=%s] %d 周期(每周期建毁1次)\n", arenaStr, cycles)
	start = getProcessPM()
	peak = start
	series = make([]float64, 0, cycles)
	for i := 0; i < cycles; i++ {
		swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOp, arenaEnabled)
		if err != nil {
			fmt.Printf("    创建 Session 失败(周期%d): %v\n", i, err)
			break
		}
		if err := runInference(swt); err != nil {
			fmt.Printf("    推理失败(周期%d): %v\n", i, err)
			destroySession(swt)
			break
		}
		destroySession(swt)
		pm := getProcessPM()
		series = append(series, pm)
		if pm > peak {
			peak = pm
		}
		if pm > 4096 {
			fmt.Printf("    [安全上限] PM 超过 4096 MB (周期%d)，停止以防 OOM 崩溃\n", i)
			break
		}
	}
	end = getProcessPM()
	return
}

// testPool: 预建1个 Session 复用 模式
func testPool(modelPath string, inputData []byte, inputShape, outputShape []int64,
	inferences, intraOp int, arenaEnabled bool) (start, peak, end float64) {
	arenaStr := "ON"
	if !arenaEnabled {
		arenaStr = "OFF"
	}
	fmt.Printf("  [pool(复用) arena=%s] %d 次推理(单Session复用)\n", arenaStr, inferences)
	swt, err := createSessionWithTensorsArena(modelPath, inputData, inputShape, outputShape, intraOp, arenaEnabled)
	if err != nil {
		fmt.Printf("    创建 Session 失败: %v\n", err)
		return 0, 0, 0
	}
	defer destroySession(swt)
	start = getProcessPM()
	peak = start
	for i := 0; i < inferences; i++ {
		if err := runInference(swt); err != nil {
			fmt.Printf("    推理失败(%d): %v\n", i, err)
			break
		}
		pm := getProcessPM()
		if pm > peak {
			peak = pm
		}
		if pm > 4096 {
			fmt.Printf("    [安全上限] PM 超过 4096 MB (第%d次推理)，停止以防 OOM 崩溃\n", i)
			break
		}
	}
	end = getProcessPM()
	return
}

func main() {
	subFlag := flag.String("sub", "", "子进程模式: per-request-off|per-request-on|pool-off|pool-on")
	flag.Parse()

	basePath, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(basePath, "third_party")); err == nil {
			break
		}
		parent := filepath.Dir(basePath)
		if parent == basePath {
			break
		}
		basePath = parent
	}
	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	modelPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	cycles := 500
	intraOp := 1
	inputShape := []int64{1, 3, 640, 640}
	outputShape := []int64{1, 84, 8400}

	// --- 子进程模式：跑单一测试，输出解析行后退出 ---
	if *subFlag != "" {
		for _, p := range []string{libPath, modelPath, inputDataPath} {
			if _, err := os.Stat(p); os.IsNotExist(err) {
				fmt.Printf("RESULT:ERROR:文件不存在:%s\n", p)
				os.Exit(1)
			}
		}
		ort.SetSharedLibraryPath(libPath)
		if err := ort.InitializeEnvironment(); err != nil {
			fmt.Printf("RESULT:ERROR:ORT初始化失败:%v\n", err)
			os.Exit(1)
		}
		defer ort.DestroyEnvironment()
		inputData, err := os.ReadFile(inputDataPath)
		if err != nil {
			fmt.Printf("RESULT:ERROR:读取输入失败:%v\n", err)
			os.Exit(1)
		}
		runtime.GOMAXPROCS(6)

		switch *subFlag {
		case "per-request-off":
			_, st, pk, en := testPerRequest(modelPath, inputData, inputShape, outputShape, cycles, intraOp, false)
			fmt.Printf("RESULT:per-request|OFF|%.2f|%.2f|%.2f|%.2f\n", st, pk, en, en-st)
		case "per-request-on":
			_, st, pk, en := testPerRequest(modelPath, inputData, inputShape, outputShape, cycles, intraOp, true)
			fmt.Printf("RESULT:per-request|ON|%.2f|%.2f|%.2f|%.2f\n", st, pk, en, en-st)
		case "pool-off":
			st, pk, en := testPool(modelPath, inputData, inputShape, outputShape, cycles, intraOp, false)
			fmt.Printf("RESULT:pool|OFF|%.2f|%.2f|%.2f|%.2f\n", st, pk, en, en-st)
		case "pool-on":
			st, pk, en := testPool(modelPath, inputData, inputShape, outputShape, cycles, intraOp, true)
			fmt.Printf("RESULT:pool|ON|%.2f|%.2f|%.2f|%.2f\n", st, pk, en, en-st)
		default:
			fmt.Printf("RESULT:ERROR:未知子进程模式:%s\n", *subFlag)
			os.Exit(1)
		}
		return
	}

	// --- 主进程：依次启动 4 个独立子进程（每个冷启动） ---
	exePath, _ := os.Executable()
	fmt.Printf("项目根: %s\n库: %s\n模型: %s\n输入: %s\n", basePath, libPath, modelPath, inputDataPath)
	fmt.Println("===== Session 生命周期内存累积复现（Go 侧对照） =====")
	fmt.Printf("模型: YOLO11x, 周期数: %d, intra_op=%d, inter_op=1, GOMAXPROCS=6\n", cycles, intraOp)
	fmt.Println("（每个 per-request 测试独立子进程冷启动，消除顺序混杂）")

	type SubResult struct {
		mode           string
		arena          string
		start, peak, end, drift float64
		series         []float64
	}
	var results []SubResult

	subTests := []struct{ mode, arena, subFlag string }{
		{"per-request", "OFF", "per-request-off"},
		{"per-request", "ON", "per-request-on"},
		{"pool", "OFF", "pool-off"},
		{"pool", "ON", "pool-on"},
	}

	for _, t := range subTests {
		cmd := exec.Command(exePath, "--sub="+t.subFlag)
		cmd.Dir = filepath.Dir(exePath)
		out, err := cmd.Output()
		if err != nil {
			fmt.Printf("  [%s arena=%s] 子进程失败: %v\n", t.mode, t.arena, err)
			continue
		}
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "RESULT:") {
				fmt.Println(line)
				continue
			}
			line = strings.TrimPrefix(line, "RESULT:")
			parts := strings.Split(line, "|")
			if len(parts) != 6 || parts[0] != t.mode || parts[1] != t.arena {
				continue
			}
			st, _ := strconv.ParseFloat(parts[2], 64)
			pk, _ := strconv.ParseFloat(parts[3], 64)
			en, _ := strconv.ParseFloat(parts[4], 64)
			dr, _ := strconv.ParseFloat(parts[5], 64)
			results = append(results, SubResult{t.mode, t.arena, st, pk, en, dr, nil})
		}
	}

	// 打印
	fmt.Println("\n===== 结果 =====")
	fmt.Printf("%-14s %-6s %-12s %-12s %-12s %-12s\n", "模式", "Arena", "起始PM", "峰值PM", "结束PM", "漂移")
	fmt.Printf("%-14s %-6s %-12s %-12s %-12s %-12s\n", "", "", "(MB)", "(MB)", "(MB)", "(MB)")
	for _, r := range results {
		fmt.Printf("%-14s %-6s %-12.2f %-12.2f %-12.2f %-12.2f\n", r.mode, r.arena, r.start, r.peak, r.end, r.drift)
	}

	// 保存
	outDir := filepath.Join(basePath, "results")
	os.MkdirAll(outDir, 0755)
	summaryPath := filepath.Join(outDir, "repro_lifecycle_go_summary.txt")
	f, err := os.Create(summaryPath)
	if err != nil {
		fmt.Printf("创建摘要文件失败: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	fmt.Fprintln(f, "===== Go Session 生命周期内存累积复现（对照） =====")
	fmt.Fprintf(f, "模型: YOLO11x, 周期数: %d, intra_op=%d, inter_op=1, GOMAXPROCS=6\n", cycles, intraOp)
	fmt.Fprintln(f, "注: 每个 per-request 测试独立子进程冷启动，消除顺序混杂")
	for _, r := range results {
		fmt.Fprintf(f, "===== %s (arena=%s) =====\n", r.mode, r.arena)
		fmt.Fprintf(f, "  起始PM: %.5f MB\n", r.start)
		fmt.Fprintf(f, "  峰值PM: %.5f MB\n", r.peak)
		fmt.Fprintf(f, "  结束PM: %.5f MB\n", r.end)
		fmt.Fprintf(f, "  总漂移(有界): %.5f MB\n", r.drift)
		fmt.Fprintf(f, "\n")
	}

	// per-request 序列 CSV（子进程模式下逐周期序列不可用，仅保留摘要）
	seriesPath := filepath.Join(outDir, "repro_lifecycle_go_perrequest_series.csv")
	sf, err := os.Create(seriesPath)
	if err != nil {
		fmt.Printf("创建序列文件失败: %v\n", err)
		os.Exit(1)
	}
	defer sf.Close()
	fmt.Fprintln(sf, "cycle,pm_arena_off,pm_arena_on")
	fmt.Fprintln(sf, "# 注: 子进程独立冷启动模式，逐周期序列不可用，仅保留摘要")

	fmt.Printf("\n摘要: %s\n", summaryPath)
	fmt.Printf("序列: %s\n", seriesPath)
}