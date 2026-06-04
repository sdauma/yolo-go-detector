package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	logFilePrefix    = "detector_"
	logRetentionDays = 7
	logBufferSize    = 4096 // 4KB 缓冲，减少磁盘 I/O
)

// Logger 运行时日志写入器
// 同时输出到终端和按小时切分的日志文件
type Logger struct {
	mu       sync.Mutex
	file     *os.File
	writer   *bufio.Writer
	dir      string
	dateHour string
}

// NewLogger 创建日志写入器
func NewLogger(dir string) (*Logger, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("创建日志目录失败: %w", err)
	}
	l := &Logger{dir: dir}
	if err := l.rotate(); err != nil {
		return nil, err
	}
	// 启动时清理过期日志
	go l.cleanOldLogs()
	return l, nil
}

// Printf 格式化输出到终端和日志文件（与 fmt.Printf 行为一致）
func (l *Logger) Printf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)

	// 终端输出
	fmt.Print(msg)

	// 文件输出（快速路径：绝大多数情况无需切文件）
	l.mu.Lock()
	nowHour := time.Now().Format("2006-01-02_15")
	if nowHour != l.dateHour {
		l.rotateLocked()
	}
	if l.writer != nil {
		l.writer.WriteString(msg)
		// 每次消息后 flush：确保崩溃不丢日志，每轮仅 3~8 条，对性能无影响
		l.writer.Flush()
	}
	l.mu.Unlock()
}

// rotate 切换日志文件
func (l *Logger) rotate() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.rotateLocked()
}

func (l *Logger) rotateLocked() error {
	if l.file != nil {
		l.writer.Flush()
		l.file.Close()
	}

	l.dateHour = time.Now().Format("2006-01-02_15")
	filename := filepath.Join(l.dir, logFilePrefix+l.dateHour+".log")

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		l.file = nil
		l.writer = nil
		return fmt.Errorf("打开日志文件失败: %w", err)
	}

	l.file = f
	l.writer = bufio.NewWriterSize(f, logBufferSize)
	return nil
}

// cleanOldLogs 清理超过保留期的旧日志文件
func (l *Logger) cleanOldLogs() {
	entries, err := os.ReadDir(l.dir)
	if err != nil {
		return
	}

	cutoff := time.Now().Add(-logRetentionDays * 24 * time.Hour)

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), logFilePrefix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			os.Remove(filepath.Join(l.dir, entry.Name()))
		}
	}
}

// Close 关闭日志写入器
func (l *Logger) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.writer != nil {
		l.writer.Flush()
	}
	if l.file != nil {
		l.file.Close()
		l.file = nil
	}
}
