package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// OutputWriter JSONL 输出写入器
// 每轮检测结果写入一行 JSON，按日期分文件
type OutputWriter struct {
	cfg *Config
	mu  sync.Mutex

	jsonlFile *os.File
	dateStr   string
}

// NewOutputWriter 创建输出写入器
func NewOutputWriter(cfg *Config) (*OutputWriter, error) {
	if err := os.MkdirAll(cfg.Output.JSONLDir, 0755); err != nil {
		return nil, fmt.Errorf("创建 JSONL 输出目录失败: %w", err)
	}

	return &OutputWriter{
		cfg: cfg,
	}, nil
}

// WriteResult 写入单条检测结果
func (ow *OutputWriter) WriteResult(result *DetectResult) error {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	if err := ow.ensureFileLocked(); err != nil {
		return err
	}

	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("序列化结果失败: %w", err)
	}

	_, err = ow.jsonlFile.Write(append(data, '\n'))
	return err
}

// SaveAlertImage 保存告警图片
// 返回保存的路径，仅检测到告警目标时调用
func (ow *OutputWriter) SaveAlertImage(result *DetectResult, imgData []byte, pipeline *DetectionPipeline) (string, error) {
	if !pipeline.HasAlert(result) {
		return "", nil
	}

	alertDir := ow.cfg.Output.AlertImageDir
	if err := os.MkdirAll(alertDir, 0755); err != nil {
		return "", fmt.Errorf("创建告警图片目录失败: %w", err)
	}

	// 文件名: {channelCode}_{timestamp}.jpg
	filename := fmt.Sprintf("%s_%s.jpg",
		result.ChannelCode,
		time.Now().Format("20060102_150405"),
	)
	outputPath := filepath.Join(alertDir, filename)

	if ow.cfg.Output.DrawBoundingBoxes {
		if err := pipeline.DrawAlertImage(result, imgData, outputPath); err != nil {
			return "", err
		}
	} else {
		if err := os.WriteFile(outputPath, imgData, 0644); err != nil {
			return "", err
		}
	}

	return outputPath, nil
}

// WriteRoundSummary 写入本轮统计摘要
func (ow *OutputWriter) WriteRoundSummary(summary *RoundSummary) error {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	if err := ow.ensureFileLocked(); err != nil {
		return err
	}

	data, err := json.Marshal(summary)
	if err != nil {
		return err
	}

	_, err = ow.jsonlFile.Write(append(data, '\n'))
	return err
}

// ensureFileLocked 确保 JSONL 文件已按日期打开（需持有锁）
func (ow *OutputWriter) ensureFileLocked() error {
	today := time.Now().Format("2006-01-02")
	if ow.jsonlFile != nil && ow.dateStr == today {
		return nil
	}

	if ow.jsonlFile != nil {
		ow.jsonlFile.Close()
	}

	filename := filepath.Join(ow.cfg.Output.JSONLDir, fmt.Sprintf("detections_%s.jsonl", today))
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("打开 JSONL 文件失败: %w", err)
	}
	ow.jsonlFile = f
	ow.dateStr = today
	return nil
}

// Close 关闭写入器
func (ow *OutputWriter) Close() {
	ow.mu.Lock()
	defer ow.mu.Unlock()
	if ow.jsonlFile != nil {
		ow.jsonlFile.Close()
		ow.jsonlFile = nil
	}
}

// RoundSummary 每轮检测汇总
type RoundSummary struct {
	Type         string `json:"type"`
	Timestamp    string `json:"timestamp"`
	TotalOnline  int    `json:"total_online"`
	SuccessCount int    `json:"success_count"`
	FailCount    int    `json:"fail_count"`
	AlertCount   int    `json:"alert_count"`
	RoundTotalMs int64  `json:"round_total_ms"`
	AvgFetchMs   int64  `json:"avg_fetch_ms"`
	AvgInferMs   int64  `json:"avg_infer_ms"`
	PoolActive   int    `json:"pool_active"`
	PoolIdle     int    `json:"pool_idle"`
}
