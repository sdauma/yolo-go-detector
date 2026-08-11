package engine

import (
	"sync"
)

type TensorMemoryPool struct {
	mu      sync.Mutex
	buffers map[int][][]float32
	maxSize int
}

func NewTensorMemoryPool(maxSize int) *TensorMemoryPool {
	return &TensorMemoryPool{
		buffers: make(map[int][][]float32),
		maxSize: maxSize,
	}
}

func (p *TensorMemoryPool) Acquire(size int) []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()

	if buffers, ok := p.buffers[size]; ok && len(buffers) > 0 {
		lastIndex := len(buffers) - 1
		buffer := buffers[lastIndex]
		p.buffers[size] = buffers[:lastIndex]
		// 复用前将长度恢复为 size（Release 时存的是 buffer[:0]，cap 仍为 size）
		return buffer[:size]
	}

	return make([]float32, size)
}

func (p *TensorMemoryPool) Release(buffer []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()

	size := cap(buffer)
	if len(p.buffers[size]) < p.maxSize {
		p.buffers[size] = append(p.buffers[size], buffer[:0])
	}
}

func (p *TensorMemoryPool) Stats() map[int]int {
	p.mu.Lock()
	defer p.mu.Unlock()

	stats := make(map[int]int)
	for size, buffers := range p.buffers {
		stats[size] = len(buffers)
	}
	return stats
}

func (p *TensorMemoryPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.buffers = make(map[int][][]float32)
}

type DetectionResult struct {
	Latency   float64
	Output    []float32
	Success    bool
	Error      error
	WorkerID   int
	Timestamp  int64
}
