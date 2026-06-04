package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/chromedp"
)

// loginViaBrowser 通过 chromedp 浏览器自动化登录
// 流程：打开登录页 → 填入凭据 → 点击登录 → 拦截 Token
// 优先从网络请求拦截 token，回退到 localStorage/sessionStorage
func (cm *CameraManager) loginViaBrowser(username, password string) (string, error) {
	log.Println("[chromedp] 启动无头浏览器进行登录...")

	// 创建浏览器上下文（忽略证书错误 + 无头模式）
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("ignore-certificate-errors", "1"),
		chromedp.Flag("headless", "new"),
		chromedp.Flag("disable-gpu", ""),
		chromedp.Flag("no-sandbox", ""),
		chromedp.Flag("disable-dev-shm-usage", ""),
	)

	allocCtx, cancelAlloc := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancelAlloc()

	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx,
		// 过滤 chromedp 内部日志：屏蔽新版 Chrome 不兼容的协议字段错误
		chromedp.WithErrorf(chromedpLogFilter),
		chromedp.WithLogf(chromedpLogFilter),
	)
	defer cancelBrowser()

	// 设置超时
	ctx, cancelTimeout := context.WithTimeout(browserCtx, 120*time.Second)
	defer cancelTimeout()

	baseURL := cm.cfg.VideoAPI.BaseURL
	var capturedToken string

	// 监听网络响应，拦截 login API 返回的 token
	chromedp.ListenTarget(ctx, func(ev interface{}) {
		if evt, ok := ev.(*network.EventResponseReceived); ok {
			if evt.Response.URL != "" && containsStr(evt.Response.URL, "/api/bms/1.0/login") {
				// 捕获到 login API 响应，异步读取 body 获取 token
				go func(reqID network.RequestID) {
					body, err := network.GetResponseBody(reqID).Do(ctx)
					if err != nil {
						return
					}
					// 手动解析 JSON 提取 token
					token := extractTokenFromJSON(string(body))
					if token != "" {
						capturedToken = token
						log.Println("[chromedp] 从网络请求拦截到 Token")
					}
				}(evt.RequestID)
			}
		}
	})

	// 执行登录操作
	err := chromedp.Run(ctx,
		network.Enable(),
		// 导航到应用首页（会自动跳转登录页）
		chromedp.Navigate(baseURL),
		// 等待登录表单加载
		chromedp.WaitVisible(`input[type="text"],input[type="password"]`, chromedp.ByQueryAll),
		chromedp.Sleep(1*time.Second),
		// 填入用户名
		chromedp.SendKeys(`input[type="text"]`, username, chromedp.ByQuery),
		// 填入密码
		chromedp.SendKeys(`input[type="password"]`, password, chromedp.ByQuery),
		// 点击登录按钮（尝试多种选择器）
		chromedp.Click(`button[type="submit"],button`, chromedp.ByQueryAll),
		// 等待登录完成（页面跳转或 token 被拦截）
		chromedp.Sleep(5*time.Second),
	)

	if err != nil {
		return "", fmt.Errorf("浏览器登录操作失败: %w", err)
	}

	// 如果从网络请求中截获了 token，直接返回
	if capturedToken != "" {
		return capturedToken, nil
	}

	// 回退：尝试从 localStorage/sessionStorage 提取 token
	var localToken string
	err = chromedp.Run(ctx,
		chromedp.Evaluate(`localStorage.getItem('token') || localStorage.getItem('access_token') || 
			sessionStorage.getItem('token') || sessionStorage.getItem('access_token') || 
			document.cookie.split(';').find(c => c.trim().startsWith('token='))?.split('=')[1] || ''`,
			&localToken,
		),
	)
	if err != nil {
		return "", fmt.Errorf("提取 Token 失败: %w", err)
	}

	if localToken != "" {
		log.Println("[chromedp] 从 localStorage/sessionStorage 提取到 Token")
		return localToken, nil
	}

	return "", fmt.Errorf("浏览器登录完成但未能提取 Token（请检查登录页面结构或手动 SetToken）")
}

// extractTokenFromJSON 从 JSON 响应体中提取 token 字段
func extractTokenFromJSON(body string) string {
	// 简单的手工解析，避免导入 encoding/json 到独立文件（已在 camera.go 中导入）
	// 查找 "token":"..." 模式
	marker := `"token":"`
	idx := 0
	for i := 0; i < len(body)-len(marker); i++ {
		if body[i:i+len(marker)] == marker {
			idx = i + len(marker)
			break
		}
	}
	if idx == 0 {
		return ""
	}

	// 提取到下一个 "
	end := idx
	for end < len(body) && body[end] != '"' {
		end++
	}
	return body[idx:end]
}

// containsStr 检查字符串 s 是否包含 substr
func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && searchStr(s, substr) >= 0
}

func searchStr(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// chromedpLogFilter 过滤 chromedp 内部日志
// 抑制新版 Chrome 与旧版 cdproto 库之间的协议兼容性错误（IPAddressSpace 等方案枚举值）
// 真正的错误由 chromedp.Run() 返回值体现，不需要重复打印
var chromedpSilencedErrors = []string{
	"IPAddressSpace",
	"could not unmarshal event",
}

func chromedpLogFilter(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	for _, keyword := range chromedpSilencedErrors {
		if strings.Contains(msg, keyword) {
			return // 抑制已知的无害协议兼容性错误
		}
	}
	// 其他 chromedp 内部日志仍然输出
	log.Printf("[chromedp] %s", msg)
}
