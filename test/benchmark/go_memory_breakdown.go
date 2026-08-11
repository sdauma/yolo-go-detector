// go_memory_breakdown.go
// Go 内存分解测试
//
// 技术说明：
// - 使用 Go AdvancedSession 接口（NewAdvancedSession），传入 opts 配置 intraOp=8, interOp=1
// - 通过传入输入/输出 Tensor 自动启用 I/O Binding
//
// 测试目的：
// - 逐步测量 Runtime 基础内存、模型权重内存、输入/输出 Tensor 内存、中间缓冲区、Session 管理开销
// - 计算总内存和峰值 RSS
// - 输出 JSON 格式结果

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// MemoryBreakdown 内存分解结构
type MemoryBreakdown struct {
	Model             string  `json:"model"`
	RuntimeBaseMB     float64 `json:"runtime_base_mb"`     // Go runtime 基础占用
	ModelWeightsMB    float64 `json:"model_weights_mb"`    // 模型权重内存
	InputTensorMB     float64 `json:"input_tensor_mb"`     // 输入张量
	OutputTensorMB    float64 `json:"output_tensor_mb"`    // 输出张量
	IntermediateMB    float64 `json:"intermediate_mb"`     // 中间缓冲区
	SessionOverheadMB float64 `json:"session_overhead_mb"` // Session管理开销
	TotalMB           float64 `json:"total_mb"`            // 总内存
	PeakRSSMB         float64 `json:"peak_rss_mb"`         // 峰值PM
}

// TestResult 统一测试结果
type TestResult struct {
	TestName  string           `json:"test_name"`
	Language  string           `json:"language"`
	YOLO11x   *MemoryBreakdown `json:"yolo11x"`
	YOLO11n   *MemoryBreakdown `json:"yolo11n"`
	Timestamp string           `json:"timestamp"`
}

// 获取RSS内存（MB）
func getRSSMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.HeapAlloc) / 1024 / 1024
}

// 强制GC并等待完成
func forceGC() {
	runtime.GC()
	runtime.Gosched()
}

// 测量模型内存分解
func measureMemoryBreakdown(modelPath string, inputData []byte, inputShape, outputShape []int64, modelName string) (*MemoryBreakdown, error) {
	fmt.Printf("\n===== 测量 %s 内存分解 =====\n", modelName)

	// 步骤1：测量 runtime 基础内存
	forceGC()
	runtimeBase := getRSSMB()
	fmt.Printf("1. Runtime 基础内存: %.2f MB\n", runtimeBase)

	// 步骤2：创建Session选项
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建会话选项失败: %v", err)
	}
	opts.SetIntraOpNumThreads(8)
	opts.SetInterOpNumThreads(1)

	// 步骤3：加载模型，测量模型权重内存
	forceGC()
	beforeModel := getRSSMB()

	// 创建输入输出张量（用于AdvancedSession）
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("创建输入Tensor失败: %v", err)
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
		opts.Destroy()
		return nil, fmt.Errorf("创建输出Tensor失败: %v", err)
	}

	// 创建Session
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
		opts.Destroy()
		return nil, fmt.Errorf("创建Session失败: %v", err)
	}

	forceGC()
	afterModel := getRSSMB()
	modelWeights := afterModel - beforeModel
	fmt.Printf("2. 模型权重内存: %.2f MB (加载后: %.2f MB)\n", modelWeights, afterModel)

	// 步骤4：测量张量内存
	inputTensorSize := float64(len(inputTensor.GetData())*4) / 1024 / 1024 // float32 = 4 bytes
	outputTensorSize := float64(len(outputTensor.GetData())*4) / 1024 / 1024
	fmt.Printf("3. 输入张量: %.2f MB, 输出张量: %.2f MB\n", inputTensorSize, outputTensorSize)

	// 步骤5：执行推理，测量中间缓冲区
	forceGC()
	beforeInference := getRSSMB()

	err = session.Run()
	if err != nil {
		session.Destroy()
		inputTensor.Destroy()
		outputTensor.Destroy()
		opts.Destroy()
		return nil, fmt.Errorf("推理失败: %v", err)
	}

	forceGC()
	afterInference := getRSSMB()
	intermediate := afterInference - beforeInference
	fmt.Printf("4. 中间缓冲区: %.2f MB (推理后: %.2f MB)\n", intermediate, afterInference)

	// 计算Session开销
	peakRSS := afterInference
	sessionOverhead := peakRSS - runtimeBase - modelWeights - inputTensorSize - outputTensorSize - intermediate
	if sessionOverhead < 0 {
		sessionOverhead = 0
	}

	// 清理
	session.Destroy()
	inputTensor.Destroy()
	outputTensor.Destroy()
	opts.Destroy()
	forceGC()

	afterCleanup := getRSSMB()
	fmt.Printf("5. 清理后内存: %.2f MB\n", afterCleanup)

	breakdown := &MemoryBreakdown{
		Model:             modelName,
		RuntimeBaseMB:     runtimeBase,
		ModelWeightsMB:    modelWeights,
		InputTensorMB:     inputTensorSize,
		OutputTensorMB:    outputTensorSize,
		IntermediateMB:    intermediate,
		SessionOverheadMB: sessionOverhead,
		TotalMB:           runtimeBase + modelWeights + inputTensorSize + outputTensorSize + intermediate + sessionOverhead,
		PeakRSSMB:         peakRSS,
	}

	fmt.Printf("\n内存分解汇总:\n")
	fmt.Printf("  Runtime 基础: %.2f MB\n", breakdown.RuntimeBaseMB)
	fmt.Printf("  模型权重: %.2f MB\n", breakdown.ModelWeightsMB)
	fmt.Printf("  输入张量: %.2f MB\n", breakdown.InputTensorMB)
	fmt.Printf("  输出张量: %.2f MB\n", breakdown.OutputTensorMB)
	fmt.Printf("  中间缓冲区: %.2f MB\n", breakdown.IntermediateMB)
	fmt.Printf("  Session开销: %.2f MB\n", breakdown.SessionOverheadMB)
	fmt.Printf("  总计: %.2f MB\n", breakdown.TotalMB)
	fmt.Printf("  峰值PM: %.2f MB\n", breakdown.PeakRSSMB)

	return breakdown, nil
}

func main() {
	fmt.Println("===== Go 模型内存分解测试 =====")

	wd, err := os.Getwd()
	if err != nil {
		fmt.Printf("获取当前目录失败: %v\n", err)
		os.Exit(1)
	}
	basePath := filepath.Dir(filepath.Dir(wd))

	libPath := filepath.Join(basePath, "third_party", "onnxruntime.dll")
	yolo11xPath := filepath.Join(basePath, "third_party", "yolo11x.onnx")
	yolo11nPath := filepath.Join(basePath, "third_party", "yolo11n.onnx")
	inputDataPath := filepath.Join(basePath, "test", "data", "input_data.bin")

	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("初始化环境失败: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	// 加载输入数据
	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("读取输入数据失败: %v\n", err)
		os.Exit(1)
	}

	result := &TestResult{
		TestName:  "Memory_Breakdown",
		Language:  "Go",
		Timestamp: "",
	}

	// 测试 YOLO11x
	if _, err := os.Stat(yolo11xPath); err == nil {
		breakdown11x, err := measureMemoryBreakdown(
			yolo11xPath, inputData,
			[]int64{1, 3, 640, 640}, []int64{1, 84, 8400},
			"YOLO11x",
		)
		if err != nil {
			fmt.Printf("测量 YOLO11x 失败: %v\n", err)
		} else {
			result.YOLO11x = breakdown11x
		}
	} else {
		fmt.Printf("YOLO11x 模型不存在: %v\n", yolo11xPath)
	}

	// 测试 YOLO11n
	if _, err := os.Stat(yolo11nPath); err == nil {
		breakdown11n, err := measureMemoryBreakdown(
			yolo11nPath, inputData,
			[]int64{1, 3, 640, 640}, []int64{1, 84, 8400},
			"YOLO11n",
		)
		if err != nil {
			fmt.Printf("测量 YOLO11n 失败: %v\n", err)
		} else {
			result.YOLO11n = breakdown11n
		}
	} else {
		fmt.Printf("YOLO11n 模型不存在: %v\n", yolo11nPath)
	}

	// 保存结果
	result.Timestamp = ""
	resultData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("序列化结果失败: %v\n", err)
		os.Exit(1)
	}

	resultFile := filepath.Join(basePath, "results", "go_memory_breakdown_result.json")
	err = os.WriteFile(resultFile, resultData, 0644)
	if err != nil {
		fmt.Printf("保存结果失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n===== 测试完成 =====\n")
	fmt.Printf("结果已保存到: %s\n", resultFile)

	// 打印对比
	if result.YOLO11x != nil && result.YOLO11n != nil {
		fmt.Printf("\n===== YOLO11x vs YOLO11n 内存对比 =====\n")
		fmt.Printf("模型权重: %.2f MB vs %.2f MB (节省 %.1f%%)\n",
			result.YOLO11x.ModelWeightsMB,
			result.YOLO11n.ModelWeightsMB,
			(1-result.YOLO11n.ModelWeightsMB/result.YOLO11x.ModelWeightsMB)*100)
		fmt.Printf("总内存: %.2f MB vs %.2f MB (节省 %.1f%%)\n",
			result.YOLO11x.TotalMB,
			result.YOLO11n.TotalMB,
			(1-result.YOLO11n.TotalMB/result.YOLO11x.TotalMB)*100)
	}
}
