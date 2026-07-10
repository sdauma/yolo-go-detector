// Package memutil provides reliable Windows process memory measurement
// using direct Windows API calls (no PowerShell overhead).
//
// On Windows, WorkingSet64 (RSS) is an unreliable proxy for ONNX Runtime memory
// because it only reflects pages currently resident in physical RAM, not total
// committed allocation. PrivateMemorySize64 captures the full private committed
// memory including C/C++ heap allocations made by ONNX Runtime.
//
// IMPORTANT: All tests MUST be run as compiled binaries (go build -o test.exe && ./test.exe),
// NOT via "go run", which creates a temporary compiler process that breaks
// os.Getpid()-based process identification.
package memutil

import (
	"syscall"
	"unsafe"
)

var (
	kernel32              = syscall.NewLazyDLL("kernel32.dll")
	procGetCurrentProcess = kernel32.NewProc("GetCurrentProcess")
	procGetProcessMemoryInfo = kernel32.NewProc("K32GetProcessMemoryInfo")
)

// PROCESS_MEMORY_COUNTERS_EX structure
type processMemoryCountersEx struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uint64
	WorkingSetSize             uint64
	QuotaPeakPagedPoolUsage    uint64
	QuotaPagedPoolUsage        uint64
	QuotaPeakNonPagedPoolUsage uint64
	QuotaNonPagedPoolUsage     uint64
	PagefileUsage              uint64
	PeakPagefileUsage          uint64
	PrivateUsage               uint64
}

// PrivateMemoryMB returns the process PrivateMemorySize64 in megabytes
// using direct Windows API (GetProcessMemoryInfo), no PowerShell overhead.
func PrivateMemoryMB() float64 {
	hProcess, _, _ := procGetCurrentProcess.Call()

	var pmc processMemoryCountersEx
	pmc.CB = uint32(unsafe.Sizeof(pmc))

	ret, _, _ := procGetProcessMemoryInfo.Call(
		hProcess,
		uintptr(unsafe.Pointer(&pmc)),
		uintptr(pmc.CB),
	)
	if ret == 0 {
		return 0
	}
	return float64(pmc.PrivateUsage) / (1024 * 1024)
}

// WorkingSetMB returns the process WorkingSet64 in megabytes
// using direct Windows API, no PowerShell overhead.
// Kept for diagnostic comparison only.
func WorkingSetMB() float64 {
	hProcess, _, _ := procGetCurrentProcess.Call()

	var pmc processMemoryCountersEx
	pmc.CB = uint32(unsafe.Sizeof(pmc))

	ret, _, _ := procGetProcessMemoryInfo.Call(
		hProcess,
		uintptr(unsafe.Pointer(&pmc)),
		uintptr(pmc.CB),
	)
	if ret == 0 {
		return 0
	}
	return float64(pmc.WorkingSetSize) / (1024 * 1024)
}
