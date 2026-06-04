package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config 生产系统配置
type Config struct {
	// 摄像头 API 配置
	VideoAPI VideoAPIConfig `yaml:"video_api"`

	// 检测模型配置
	Detection DetectionConfig `yaml:"detection"`

	// 调度配置
	Scheduler SchedulerConfig `yaml:"scheduler"`

	// 输出配置
	Output OutputConfig `yaml:"output"`
}

// VideoAPIConfig 视联网 API 配置
type VideoAPIConfig struct {
	BaseURL    string `yaml:"base_url"`    // API 地址，如 https://172.18.137.20:9502
	Username   string `yaml:"username"`    // 登录用户名
	Password   string `yaml:"password"`    // 登录密码
	OrgID      string `yaml:"org_id"`      // 组织 ID，空=全部
	SkipVerify bool   `yaml:"skip_verify"` // 跳过 HTTPS 证书验证（内网自签名）

	// 每 4 小时刷新一次在线摄像头列表
	RefreshIntervalMinutes int `yaml:"refresh_interval_minutes"`
}

// DetectionConfig 检测模型配置
type DetectionConfig struct {
	ModelPath  string  `yaml:"model_path"`   // ONNX 模型路径
	ONNXLibPath string `yaml:"onnx_lib_path"` // ONNX Runtime 动态库路径（空=自动搜索）
	ConfThresh float64 `yaml:"conf_thresh"`  // 置信度阈值
	IOUThresh  float64 `yaml:"iou_thresh"`   // IOU 阈值
	InputSize  int     `yaml:"input_size"`   // 模型输入尺寸
	PoolSize   int     `yaml:"pool_size"`    // Session Pool 大小（0=CPU核数）
	IntraOpThreads int `yaml:"intra_op_threads"` // 每Session线程数（0=自动计算 CPU/Pool，推荐0）
}

// SchedulerConfig 调度配置
type SchedulerConfig struct {
	// 每轮检测间隔（秒），2600路并发检测
	RoundIntervalSeconds int `yaml:"round_interval_seconds"`

	// 并发取图数（同时发出的 HTTP 请求数上限）
	FetchConcurrency int `yaml:"fetch_concurrency"`

	// 单路取图超时（秒）
	FetchTimeoutSeconds int `yaml:"fetch_timeout_seconds"`
}

// OutputConfig 输出配置
type OutputConfig struct {
	// JSONL 输出目录
	JSONLDir string `yaml:"jsonl_dir"`

	// 告警图片输出目录（仅检测到目标时存图）
	AlertImageDir string `yaml:"alert_image_dir"`

	// 运行日志输出目录（按小时切文件，自动保留 7 天）
	LogDir string `yaml:"log_dir"`

	// 是否在告警图上绘制边界框
	DrawBoundingBoxes bool `yaml:"draw_bounding_boxes"`

	// 关注的类别（检测到这些类别才触发告警存图）
	AlertClasses []string `yaml:"alert_classes"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() Config {
	return Config{
		VideoAPI: VideoAPIConfig{
			BaseURL:                "https://172.18.137.20:9502",
			SkipVerify:             true,
			RefreshIntervalMinutes: 240, // 4小时
		},
		Detection: DetectionConfig{
			ModelPath:      "../third_party/yolo11x.onnx",
			ConfThresh:     0.25,
			IOUThresh:      0.7,
			InputSize:      640,
		PoolSize:       0, // 0=自动（CPU/4，上限3；大模型推荐2）
		IntraOpThreads: 0, // 0=自动计算（CPU/PoolSize）
		},
		Scheduler: SchedulerConfig{
			RoundIntervalSeconds: 5,
			FetchConcurrency:     100,
			FetchTimeoutSeconds:  30,
		},
		Output: OutputConfig{
			JSONLDir:          "./output",
			AlertImageDir:     "./output/alerts",
			LogDir:            "./output",
			DrawBoundingBoxes: true,
			AlertClasses:      []string{"person", "car", "motorcycle", "bus", "truck"},
		},
	}
}

// LoadConfig 从 YAML 文件加载配置
func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("配置文件 %s 不存在，使用默认配置\n", path)
			return &cfg, nil
		}
		return nil, fmt.Errorf("读取配置文件失败: %w", err)
	}

	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("解析配置文件失败: %w", err)
	}

	return &cfg, nil
}
