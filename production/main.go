package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"yolo-go-detector/engine"
)

func main() {
	configPath := flag.String("config", "./config.yaml", "配置文件路径")
	flag.Parse()

	fmt.Println("============================================")
	fmt.Println("  YOLO-Go 生产级摄像头检测系统")
	fmt.Println("  基于论文 Session Pool 架构")
	fmt.Println("============================================")

	// 加载配置
	cfg, err := LoadConfig(*configPath)
	if err != nil {
		fmt.Printf("加载配置失败: %v\n", err)
		os.Exit(1)
	}

	if cfg.VideoAPI.Username == "" && cfg.VideoAPI.Password == "" {
		fmt.Println("[提示] 未配置摄像头 API 用户名/密码")
		fmt.Println("  - 如果是 chromedp 登录系统，程序将自动尝试浏览器登录")
		fmt.Println("  - 如果手动获取 Token，请通过 SetToken() 设置后程序自动就绪")
	}

	fmt.Printf("[配置] 模型: %s\n", cfg.Detection.ModelPath)
	fmt.Printf("[配置] 置信度: %.2f, IOU: %.2f, 输入尺寸: %d\n",
		cfg.Detection.ConfThresh, cfg.Detection.IOUThresh, cfg.Detection.InputSize)
	fmt.Printf("[配置] Session Pool 大小: %d\n", cfg.Detection.PoolSize)
	fmt.Printf("[配置] 每轮间隔: %d 秒, 取图并发: %d\n",
		cfg.Scheduler.RoundIntervalSeconds, cfg.Scheduler.FetchConcurrency)

	// 检查 ONNX 模型是否存在
	if _, err := os.Stat(cfg.Detection.ModelPath); os.IsNotExist(err) {
		fmt.Printf("[警告] ONNX 模型不存在: %s\n", cfg.Detection.ModelPath)
		fmt.Printf("  请确认模型文件路径正确，或通过 --config 指定正确的配置文件\n")
	}

	// 设置 ONNX Runtime 动态库路径（优先级：配置文件 > 自动搜索）
	if cfg.Detection.ONNXLibPath != "" {
		engine.SetONNXLibPath(cfg.Detection.ONNXLibPath)
		fmt.Printf("[配置] ONNX Runtime 库路径: %s (来自配置文件)\n", cfg.Detection.ONNXLibPath)
	}

	ctx := context.Background()

	// 初始化摄像头管理器
	camMgr := NewCameraManager(ctx, cfg)

	// 初始化检测流水线
	fmt.Println("[初始化] 创建 Session Pool...")
	pipeline, err := NewDetectionPipeline(cfg)
	if err != nil {
		fmt.Printf("创建检测流水线失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("[初始化] Session Pool 创建成功")

	// 初始化输出写入器
	output, err := NewOutputWriter(cfg)
	if err != nil {
		fmt.Printf("创建输出写入器失败: %v\n", err)
		os.Exit(1)
	}

	// 初始化日志写入器（终端 + 文件双写，按小时切）
	logger, err := NewLogger(cfg.Output.LogDir)
	if err != nil {
		fmt.Printf("创建日志写入器失败: %v\n", err)
		os.Exit(1)
	}
	defer logger.Close()
	fmt.Println("[初始化] 日志写入器初始化成功")

	// 创建并启动调度器
	scheduler := NewScheduler(cfg, camMgr, pipeline, output, logger)

	// 处理 Ctrl+C 信号
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\n[信号] 收到停止信号，正在优雅退出...")
		scheduler.Stop()
		os.Exit(0)
	}()

	// 启动调度循环（会阻塞）
	scheduler.Run()
}
