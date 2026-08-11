// go_session_pool_fault_injection.go
// Session Pool 故障注入实验（Fault Injection Experiment）
//
// 目的：受控验证 engine.SessionPool 的"Session 级故障隔离 + 自动重建"能力
// （论文 §4.6.1 / §5.1 / §5.2 所述架构价值），无需以长期压测间接佐证。
//
// 设计：
//   - 自变量：池容量 pool ∈ {2, 4}
//   - 故障注入：从池中取出全部会话后，对其中 1 个会话注入故障
//     （Destroy() 模拟 ONNX 句柄失效，再 PutSessionBroken() 触发故障处理路径）
//   - 验证项：
//       (a) 故障隔离：其余 N-1 个会话 Run() 仍成功（单点故障不波及其他会话）
//       (b) 自动重建：GetSession() 在 activeSessions < maxSize 时应惰性创建替代实例
//       (c) 重建可用性：新会话 Run() 成功（池容量恢复至设定值）
//   - 每配置重复 rounds 轮（默认 20），全轮通过方判定该配置验证成功
//   - 控制：YOLO11x、inputSize=640、batchSize=1、intraOp=6、GOMAXPROCS=6
//
// 预期（供论文核对）：隔离与重建应全部成功——这把"架构性质、未实测"升级为
// "经故障注入实验验证"。若实测失败，则如实反映（不应掩盖）。
//
// 注意：本程序复用 engine.SessionPool 的生产级故障处理 API
// （PutSessionBroken + GetSession 惰性重建），非新增代码路径。
// NewSessionPool 内部已自动初始化 ORT 环境与预创建会话。

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
	"yolo-go-detector/engine"
	"yolo-go-detector/test/benchmark/memutil"
)

func findRoot() string {
	basePath, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(basePath, "third_party")); err == nil {
			return basePath
		}
		parent := filepath.Dir(basePath)
		if parent == basePath {
			break
		}
		basePath = parent
	}
	return basePath
}

func fillInput(input *ort.Tensor[float32], inputData []byte) {
	floatData := input.GetData()
	for j := 0; j < len(floatData); j++ {
		if j*4 < len(inputData) {
			bits := binary.LittleEndian.Uint32(inputData[j*4 : j*4+4])
			floatData[j] = math.Float32frombits(bits)
		}
	}
}

// injectFault: 取满池 → 注入 1 个会话故障 → 验证隔离 + 重建 + 恢复
// 返回 (隔离是否成功, 重建是否成功, 恢复是否成功, 归还后活跃会话数)。
// 注意：activeSessions 计数的是"已取出未归还"的会话数，全部 PutSession 归还后归零（会话回到空闲通道），故期望值恒为 0，与池容量无关。
func injectFault(pool *engine.SessionPool, inputData []byte, poolSize int) (isolationOK, rebuildOK, recoverOK bool, activeAfter int) {
	// 1. 取满池（全部 N 个会话均被取出，空闲通道为空）
	sessions := make([]*engine.ModelSession, 0, poolSize)
	for i := 0; i < poolSize; i++ {
		s, err := pool.GetSession()
		if err != nil {
			fmt.Printf("    [ERROR] 取会话失败(第%d个): %v\n", i, err)
			for _, s2 := range sessions {
				pool.PutSession(s2)
			}
			return false, false, false, 0
		}
		sessions = append(sessions, s)
	}

	// 2. 注入故障：销毁第 0 个会话的 ONNX 句柄，再走 PutSessionBroken
	victim := sessions[0]
	sessions = sessions[1:] // 其余 N-1 个保留在手中（模拟在线会话）
	if victim.Session != nil {
		victim.Session.Destroy() // 模拟 ONNX 句柄失效
	}
	pool.PutSessionBroken(victim) // 触发故障处理：销毁而非归还，activeSessions 减 1

	// 3. 验证隔离：其余 N-1 个会话 Run() 应仍成功
	isolationOK = true
	for idx, s := range sessions {
		fillInput(s.Input, inputData)
		if err := s.Session.Run(); err != nil {
			fmt.Printf("    [FAIL] 隔离验证：第%d个未故障会话 Run() 失败: %v\n", idx, err)
			isolationOK = false
		}
	}

	// 4. 验证重建：GetSession() 应惰性创建替代实例（activeSessions 恢复到 N）
	newSession, err := pool.GetSession()
	if err != nil {
		fmt.Printf("    [FAIL] 重建验证：GetSession() 失败: %v\n", err)
		rebuildOK = false
	} else {
		rebuildOK = true
		// 5. 验证恢复：新会话 Run() 应成功
		fillInput(newSession.Input, inputData)
		if err := newSession.Session.Run(); err != nil {
			fmt.Printf("    [FAIL] 恢复验证：新会话 Run() 失败: %v\n", err)
			recoverOK = false
		} else {
			recoverOK = true
		}
		pool.PutSession(newSession)
	}

	// 6. 归还其余会话，检查池容量恢复
	for _, s := range sessions {
		pool.PutSession(s)
	}
	active, _ := pool.GetStats()
	activeAfter = active // 全部归还后应与 poolSize 一致（池满）

	return isolationOK, rebuildOK, recoverOK, activeAfter
}

func runConfig(poolSize, rounds int, modelPath string, inputData []byte) (passed, total, lastActive int) {
	fmt.Printf("\n  ===== pool=%d, 重复 %d 轮 =====\n", poolSize, rounds)
	pool := engine.NewSessionPool(poolSize, modelPath, 640, 1, 6)
	passed = 0
	total = rounds
	for r := 0; r < rounds; r++ {
		iso, reb, rec, act := injectFault(pool, inputData, poolSize)
		ok := iso && reb && rec
		if ok {
			passed++
		}
		fmt.Printf("    轮 %2d/%d: 隔离=%v 重建=%v 恢复=%v 归还后active=%d (期望0,会话已全归还) -> %s\n",
			r+1, rounds, iso, reb, rec, act, boolToStatus(ok))
		lastActive = act
	}
	return passed, total, lastActive
}

func boolToStatus(b bool) string {
	if b {
		return "PASS"
	}
	return "FAIL"
}

func main() {
	roundsFlag := flag.Int("rounds", 20, "每组配置重复轮数")
	flag.Parse()

	root := findRoot()
	modelPath := filepath.Join(root, "third_party", "yolo11x.onnx")
	inputDataPath := filepath.Join(root, "test", "data", "input_data.bin")

	for _, p := range []string{modelPath, inputDataPath} {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			fmt.Printf("RESULT:ERROR:文件不存在:%s\n", p)
			os.Exit(1)
		}
	}

	inputData, err := os.ReadFile(inputDataPath)
	if err != nil {
		fmt.Printf("RESULT:ERROR:读取输入失败:%v\n", err)
		os.Exit(1)
	}
	runtime.GOMAXPROCS(6)

	fmt.Println("===== Session Pool 故障注入实验（隔离 + 自动重建验证） =====")
	fmt.Printf("模型: YOLO11x, 每组轮数: %d, inputSize=640, batchSize=1, intraOp=6, GOMAXPROCS=6\n", *roundsFlag)
	fmt.Println("验证项: (a) 故障隔离 (b) 自动重建 (c) 重建后可用性")
	fmt.Printf("进程私有内存基线: %.1f MB\n", memutil.PrivateMemoryMB())

	// 结果文件：已存在则不覆盖（由 bat skip 逻辑处理；此处仅在不存在时写）
	outDir := filepath.Join(root, "results")
	os.MkdirAll(outDir, 0755)
	resultPath := filepath.Join(outDir, "go_session_pool_fault_injection_result.txt")
	if _, err := os.Stat(resultPath); err == nil {
		fmt.Printf("RESULT:SKIP:结果文件已存在，未覆盖: %s\n", resultPath)
		fmt.Printf("（如需重跑，请先移走或归档该文件）\n")
		return
	}

	p2Pass, p2Total, p2Active := runConfig(2, *roundsFlag, modelPath, inputData)
	p4Pass, p4Total, p4Active := runConfig(4, *roundsFlag, modelPath, inputData)

	overallOK := p2Pass == p2Total && p4Pass == p4Total

	// 写结果
	f, err := os.Create(resultPath)
	if err != nil {
		fmt.Printf("RESULT:ERROR:创建结果文件失败:%v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	fmt.Fprintf(f, "===== Session Pool 故障注入实验 =====\n")
	fmt.Fprintf(f, "模型: YOLO11x, 每组轮数: %d, inputSize=640, batchSize=1, intraOp=6, GOMAXPROCS=6\n", *roundsFlag)
	fmt.Fprintf(f, "验证项: (a) 故障隔离 (b) 自动重建 (c) 重建后可用性\n\n")
	fmt.Fprintf(f, "--- pool=2 ---\n")
	fmt.Fprintf(f, "  通过轮数: %d / %d\n", p2Pass, p2Total)
	fmt.Fprintf(f, "  归还后活跃会话数(末轮): %d (期望 0,会话已全部归还空闲通道)\n", p2Active)
	fmt.Fprintf(f, "--- pool=4 ---\n")
	fmt.Fprintf(f, "  通过轮数: %d / %d\n", p4Pass, p4Total)
	fmt.Fprintf(f, "  归还后活跃会话数(末轮): %d (期望 0,会话已全部归还空闲通道)\n", p4Active)
	fmt.Fprintf(f, "\n总体结论: %s\n", boolToStatus(overallOK))
	if overallOK {
		fmt.Fprintf(f, "说明: Session 级故障隔离与自动重建经故障注入实验验证通过；\n")
		fmt.Fprintf(f, "      单点会话故障不波及其他会话，且池在容量上限内惰性重建替代实例。\n")
	} else {
		fmt.Fprintf(f, "说明: 存在失败轮次，请检查 engine.SessionPool 故障处理链路。\n")
	}

	fmt.Printf("\n===== 结果 =====\n")
	fmt.Printf("pool=2: %d/%d 通过, 末轮归还后active=%d\n", p2Pass, p2Total, p2Active)
	fmt.Printf("pool=4: %d/%d 通过, 末轮归还后active=%d\n", p4Pass, p4Total, p4Active)
	fmt.Printf("总体: %s\n", boolToStatus(overallOK))
	fmt.Printf("结果文件: %s\n", resultPath)
	fmt.Printf("RESULT:%s\n", boolToStatus(overallOK))
}
