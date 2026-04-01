package engine

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

type PerformanceMonitor struct {
	mu                sync.Mutex
	latencyHistory    []float64
	throughputHistory []float64
	errorCount        int
	totalRequests     int
	startTime         time.Time
}

func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		latencyHistory:    make([]float64, 0, 1000),
		throughputHistory: make([]float64, 0, 100),
		startTime:         time.Now(),
	}
}

func (m *PerformanceMonitor) RecordLatency(latency float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.latencyHistory = append(m.latencyHistory, latency)
	m.totalRequests++

	if len(m.latencyHistory) > 1000 {
		m.latencyHistory = m.latencyHistory[1:]
	}
}

func (m *PerformanceMonitor) RecordError() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errorCount++
}

func (m *PerformanceMonitor) RecordThroughput(throughput float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.throughputHistory = append(m.throughputHistory, throughput)
	if len(m.throughputHistory) > 100 {
		m.throughputHistory = m.throughputHistory[1:]
	}
}

func (m *PerformanceMonitor) GetAverageLatency() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.latencyHistory) == 0 {
		return 0
	}

	var sum float64
	for _, lat := range m.latencyHistory {
		sum += lat
	}
	return sum / float64(len(m.latencyHistory))
}

func (m *PerformanceMonitor) GetP50Latency() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.latencyHistory) == 0 {
		return 0
	}

	sorted := make([]float64, len(m.latencyHistory))
	copy(sorted, m.latencyHistory)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted[int(float64(len(sorted))*0.5)]
}

func (m *PerformanceMonitor) GetP90Latency() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.latencyHistory) == 0 {
		return 0
	}

	sorted := make([]float64, len(m.latencyHistory))
	copy(sorted, m.latencyHistory)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted[int(float64(len(sorted))*0.9)]
}

func (m *PerformanceMonitor) GetP99Latency() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.latencyHistory) == 0 {
		return 0
	}

	sorted := make([]float64, len(m.latencyHistory))
	copy(sorted, m.latencyHistory)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted[int(float64(len(sorted))*0.99)]
}

func (m *PerformanceMonitor) GetOptimalThreadCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.latencyHistory) < 100 {
		return runtime.NumCPU()
	}

	avgLatency := m.GetAverageLatency()

	if avgLatency < 50 {
		return runtime.NumCPU()
	} else if avgLatency < 100 {
		return runtime.NumCPU() / 2
	} else {
		return 4
	}
}

func (m *PerformanceMonitor) GetErrorRate() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.totalRequests == 0 {
		return 0
	}
	return float64(m.errorCount) / float64(m.totalRequests) * 100
}

func (m *PerformanceMonitor) GetStats() PerformanceStats {
	m.mu.Lock()
	defer m.mu.Unlock()

	return PerformanceStats{
		AverageLatency: m.GetAverageLatency(),
		P50Latency:     m.GetP50Latency(),
		P90Latency:     m.GetP90Latency(),
		P99Latency:     m.GetP99Latency(),
		ErrorRate:      m.GetErrorRate(),
		TotalRequests:  m.totalRequests,
		ErrorCount:     m.errorCount,
	}
}

type PerformanceStats struct {
	AverageLatency float64
	P50Latency     float64
	P90Latency     float64
	P99Latency     float64
	ErrorRate      float64
	TotalRequests  int
	ErrorCount     int
}

func (s PerformanceStats) String() string {
	return "PerformanceStats{\n" +
		"  AverageLatency: " + fmt.Sprintf("%.2f", s.AverageLatency) + " ms\n" +
		"  P50Latency: " + fmt.Sprintf("%.2f", s.P50Latency) + " ms\n" +
		"  P90Latency: " + fmt.Sprintf("%.2f", s.P90Latency) + " ms\n" +
		"  P99Latency: " + fmt.Sprintf("%.2f", s.P99Latency) + " ms\n" +
		"  ErrorRate: " + fmt.Sprintf("%.2f", s.ErrorRate) + "%\n" +
		"  TotalRequests: " + fmt.Sprintf("%d", s.TotalRequests) + "\n" +
		"  ErrorCount: " + fmt.Sprintf("%d", s.ErrorCount) + "\n" +
		"}"
}
