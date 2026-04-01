package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Println("Go 环境测试")
	fmt.Printf("Go 版本: %s\n", runtime.Version())
	fmt.Printf("默认线程数: %d\n", runtime.GOMAXPROCS(0))
	
	// 测试设置线程数
	runtime.GOMAXPROCS(12)
	fmt.Printf("设置后线程数: %d\n", runtime.GOMAXPROCS(0))
	
	fmt.Println("测试完成!")
}