package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"
)

// CameraInfo 摄像头信息
type CameraInfo struct {
	ChannelCode string `json:"channelCode"`
	ChannelName string `json:"channelName"`
	OrgName     string `json:"orgName"`
	Status      int    `json:"status"` // 0=离线, 1=在线
}

// CameraManager 摄像头管理器
// 负责：Token 管理、登录状态维护、在线列表刷新、HTTP 客户端
type CameraManager struct {
	cfg    *Config
	client *http.Client

	mu     sync.RWMutex
	online map[string]*CameraInfo // channelCode -> CameraInfo

	token   string
	tokenMu sync.RWMutex

	ctx context.Context

	// 登录状态管理
	loginReady atomic.Bool   // true = 已成功登录过（初始登录完成）
	loginDone  chan struct{} // 初始登录完成时 close
	loginErr   error         // 初始登录失败原因（loginMu 保护）
	loginMu    sync.Mutex    // 保护登录过程（避免并发登录）
}

// NewCameraManager 创建摄像头管理器
func NewCameraManager(ctx context.Context, cfg *Config) *CameraManager {
	// HTTP 连接池与并发取图数匹配，避免连接瓶颈
	maxConns := cfg.Scheduler.FetchConcurrency + 100
	if maxConns < 200 {
		maxConns = 200
	}
	maxPerHost := maxConns

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: cfg.VideoAPI.SkipVerify,
		},
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		MaxIdleConns:          maxConns,
		MaxIdleConnsPerHost:   maxPerHost,
		MaxConnsPerHost:       maxPerHost * 2, // 允许超额的活跃连接
	}

	return &CameraManager{
		cfg:       cfg,
		client:    &http.Client{Transport: tr, Timeout: 30 * time.Second},
		online:    make(map[string]*CameraInfo),
		ctx:       ctx,
		loginDone: make(chan struct{}),
	}
}

// GetOnlineCameras 获取当前在线摄像头列表（快照）
func (cm *CameraManager) GetOnlineCameras() []*CameraInfo {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	result := make([]*CameraInfo, 0, len(cm.online))
	for _, cam := range cm.online {
		result = append(result, cam)
	}
	return result
}

// OnlineCount 返回在线摄像头数量
func (cm *CameraManager) OnlineCount() int {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return len(cm.online)
}

// RefreshOnlineList 从 API 刷新在线摄像头列表
func (cm *CameraManager) RefreshOnlineList() error {
	var allCameras []CameraInfo

	for page := 0; ; page++ {
		cameras, hasMore, err := cm.fetchOnlinePage(page, 1500)
		if err != nil {
			return fmt.Errorf("获取第 %d 页在线列表失败: %w", page, err)
		}
		allCameras = append(allCameras, cameras...)
		if !hasMore || len(cameras) == 0 {
			break
		}
	}

	cm.mu.Lock()
	cm.online = make(map[string]*CameraInfo, len(allCameras))
	for i := range allCameras {
		cm.online[allCameras[i].ChannelCode] = &allCameras[i]
	}
	cm.mu.Unlock()

	log.Printf("[摄像头] 在线列表刷新完成: %d 路\n", len(allCameras))
	return nil
}

// fetchOnlinePage 分页获取在线摄像头
func (cm *CameraManager) fetchOnlinePage(page, size int) ([]CameraInfo, bool, error) {
	token, err := cm.getToken()
	if err != nil {
		return nil, false, err
	}

	params := url.Values{}
	params.Add("page", fmt.Sprintf("%d", page))
	params.Add("size", fmt.Sprintf("%d", size))
	params.Add("sort", "channelCode,asc")
	params.Add("showLeaf", "true")
	params.Add("status", "1")

	if orgID := cm.cfg.VideoAPI.OrgID; orgID != "" {
		params.Add("orgId", orgID)
	}

	reqURL := cm.cfg.VideoAPI.BaseURL + "/api/bms/1.0/square/list?" + params.Encode()
	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, false, err
	}
	cm.setHeaders(req, token)

	resp, err := cm.client.Do(req)
	if err != nil {
		return nil, false, fmt.Errorf("请求失败: %w", err)
	}
	defer resp.Body.Close()

	// 检测 Token 过期（HTTP 401/403）
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		cm.invalidateToken()
		return nil, false, fmt.Errorf("Token 已过期(HTTP %d)，已触发重新登录", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, err
	}

	var result struct {
		Success bool `json:"success"`
		Data    struct {
			Content       []CameraInfo `json:"content"`
			TotalElements int          `json:"totalElements"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, false, fmt.Errorf("JSON解析失败: %w", err)
	}

	// 检查业务层权限错误（JSON 中包含未授权标记）
	if !result.Success {
		return nil, false, fmt.Errorf("API返回失败")
	}

	total := page*size + len(result.Data.Content)
	return result.Data.Content, total < result.Data.TotalElements, nil
}

// FetchSnapshot 获取指定摄像头的实时快照
// 返回图片字节流（纯内存，不落盘）
// ctx 控制单路取图超时（受 config.fetch_timeout_seconds 约束）
func (cm *CameraManager) FetchSnapshot(ctx context.Context, channelCode string) ([]byte, error) {
	token, err := cm.getToken()
	if err != nil {
		return nil, err
	}

	// 单路取图超时
	fetchTimeout := time.Duration(cm.cfg.Scheduler.FetchTimeoutSeconds) * time.Second
	if fetchTimeout <= 0 {
		fetchTimeout = 30 * time.Second
	}
	reqCtx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()

	params := url.Values{}
	params.Add("channelCode", channelCode)
	params.Add("update", "true")
	params.Add("ifCompress", "true")
	params.Add("t", fmt.Sprintf("%d", time.Now().UnixMilli()))

	reqURL := cm.cfg.VideoAPI.BaseURL + "/api/bms/1.0/square/snapImg?" + params.Encode()
	req, err := http.NewRequestWithContext(reqCtx, "GET", reqURL, nil)
	if err != nil {
		return nil, err
	}
	cm.setHeaders(req, token)

	resp, err := cm.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("获取快照失败 [%s]: %w", channelCode, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		cm.invalidateToken()
		return nil, fmt.Errorf("Token 已过期(HTTP %d) [%s]，已触发重新登录", resp.StatusCode, channelCode)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("快照返回 %d [%s]", resp.StatusCode, channelCode)
	}

	bodyData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取快照数据失败 [%s]: %w", channelCode, err)
	}

	if len(bodyData) == 0 {
		return nil, fmt.Errorf("快照数据为空 [%s]", channelCode)
	}

	// 检查是否为 JSON 错误响应
	if bodyData[0] == '{' {
		var errResp struct {
			Code    string `json:"code"`
			Success bool   `json:"success"`
		}
		if json.Unmarshal(bodyData, &errResp) == nil && !errResp.Success {
			return nil, fmt.Errorf("API错误 [%s]: code=%s", channelCode, errResp.Code)
		}
	}

	// 检查是否为有效图片（排除默认预览图，通常 < 10KB）
	if len(bodyData) < 10240 {
		return nil, fmt.Errorf("图片过小(%d字节) [%s]", len(bodyData), channelCode)
	}

	contentType := http.DetectContentType(bodyData)
	if contentType == "" || (contentType != "" && contentType[:6] != "image/") {
		return nil, fmt.Errorf("非图片数据 [%s]: %s", channelCode, contentType)
	}

	return bodyData, nil
}

// getToken 获取认证 Token（登录守卫：确保已登录后才返回 Token）
// 如果 token 为空则等待登录完成；如果检测到 token 过期则触发重登
func (cm *CameraManager) getToken() (string, error) {
	// 快速路径：已有 token，直接返回
	cm.tokenMu.RLock()
	token := cm.token
	cm.tokenMu.RUnlock()
	if token != "" {
		return token, nil
	}

	// 等待初始登录完成
	if err := cm.WaitReady(cm.ctx); err != nil {
		return "", err
	}

	// 再次尝试获取 token
	cm.tokenMu.RLock()
	token = cm.token
	cm.tokenMu.RUnlock()
	if token != "" {
		return token, nil
	}

	return cm.doLogin()
}

// doLogin 执行实际登录（API → chromedp 回退）
func (cm *CameraManager) doLogin() (string, error) {
	cm.loginMu.Lock()
	defer cm.loginMu.Unlock()

	// 双重检查
	cm.tokenMu.RLock()
	if cm.token != "" {
		cm.tokenMu.RUnlock()
		return cm.token, nil
	}
	cm.tokenMu.RUnlock()

	username := cm.cfg.VideoAPI.Username
	password := cm.cfg.VideoAPI.Password

	if username == "" || password == "" {
		return "", fmt.Errorf("未配置摄像头 API 用户名/密码，请通过 SetToken() 手动设置")
	}

	// 1. 尝试 API 登录（快）
	token, err := cm.loginViaAPI(username, password)
	if err == nil {
		cm.tokenMu.Lock()
		cm.token = token
		cm.tokenMu.Unlock()
		cm.markLoginReady()
		log.Println("[摄像头] API 登录成功，Token 已就绪")
		return token, nil
	}
	log.Printf("[摄像头] API 登录失败: %v，尝试 chromedp 浏览器登录...\n", err)

	// 2. 回退到 chromedp 浏览器登录（慢）
	apiErr := err // 保存 API 登录错误，避免被浏览器错误覆盖
	token, err = cm.loginViaBrowser(username, password)
	if err != nil {
		cm.loginErr = fmt.Errorf("所有登录方式均失败: API(%v), 浏览器(%v)", apiErr, err)
		return "", cm.loginErr
	}

	cm.tokenMu.Lock()
	cm.token = token
	cm.tokenMu.Unlock()
	cm.markLoginReady()
	log.Println("[摄像头] chromedp 浏览器登录成功，Token 已就绪")
	return token, nil
}

// invalidateToken 使当前 token 失效，触发下一轮重新登录
// 当 API 返回 401/权限错误时调用
func (cm *CameraManager) invalidateToken() {
	cm.tokenMu.Lock()
	cm.token = ""
	cm.tokenMu.Unlock()
	log.Println("[摄像头] Token 已失效，将在下一轮自动重新登录")
}

// markLoginReady 标记登录就绪
func (cm *CameraManager) markLoginReady() {
	if cm.loginReady.CompareAndSwap(false, true) {
		close(cm.loginDone)
	}
	cm.loginErr = nil
}

// WaitReady 等待登录就绪（阻塞直到首次登录成功或 ctx 被取消）
func (cm *CameraManager) WaitReady(ctx context.Context) error {
	if cm.loginReady.Load() {
		return nil
	}
	select {
	case <-cm.loginDone:
		if cm.loginErr != nil {
			return cm.loginErr
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Login 执行登录并阻塞直到成功
// 调用方（scheduler）在启动前调用，确保登录完成后才开始调度
func (cm *CameraManager) Login(ctx context.Context) error {
	// 如果已登录，直接返回
	if cm.loginReady.Load() {
		return nil
	}

	username := cm.cfg.VideoAPI.Username
	password := cm.cfg.VideoAPI.Password

	if username == "" || password == "" {
		// 无凭据，等待外部 SetToken() 调用
		log.Println("[摄像头] 未配置用户名/密码，等待外部 SetToken() 设置 Token...")
		return cm.WaitReady(ctx)
	}

	// 执行登录
	_, err := cm.doLogin()
	if err != nil {
		return err
	}
	return nil
}

// IsLoggedIn 检查当前是否处于已登录状态
func (cm *CameraManager) IsLoggedIn() bool {
	cm.tokenMu.RLock()
	defer cm.tokenMu.RUnlock()
	return cm.token != ""
}

// loginViaAPI 通过 API JSON 登录
func (cm *CameraManager) loginViaAPI(username, password string) (string, error) {
	reqBody, _ := json.Marshal(map[string]string{
		"username": username,
		"password": password,
	})

	reqURL := cm.cfg.VideoAPI.BaseURL + "/api/bms/1.0/login"
	req, err := http.NewRequest("POST", reqURL, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json;charset=UTF-8")
	req.Header.Set("Accept", "application/json, text/plain, */*")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := cm.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("登录请求失败: %w", err)
	}
	defer resp.Body.Close()

	var loginResp struct {
		Code    string `json:"code"`
		Success bool   `json:"success"`
		Data    struct {
			Token string `json:"token"`
		} `json:"data"`
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if err := json.Unmarshal(body, &loginResp); err != nil {
		return "", fmt.Errorf("解析登录响应失败: %w", err)
	}

	if !loginResp.Success || loginResp.Data.Token == "" {
		return "", fmt.Errorf("登录失败: code=%s, 可能需要 chromedp 浏览器登录", loginResp.Code)
	}

	return loginResp.Data.Token, nil
}

// setHeaders 设置通用请求头
func (cm *CameraManager) setHeaders(req *http.Request, token string) {
	req.Header.Set("Accept", "application/json, text/plain, */*")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Pragma", "no-cache")
	req.Header.Set("Content-Type", "application/json;charset=UTF-8")
	req.Header.Set("Referer", cm.cfg.VideoAPI.BaseURL+"/")
	req.Header.Set("Authorization", token)
	req.Header.Set("sec-ch-ua", `"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"`)
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-platform", `"Windows"`)
	req.Header.Set("Sec-Fetch-Site", "same-origin")
	req.Header.Set("Sec-Fetch-Mode", "cors")
	req.Header.Set("Sec-Fetch-Dest", "empty")
	req.Header.Set("Connection", "keep-alive")
	if u, err := url.Parse(cm.cfg.VideoAPI.BaseURL); err == nil {
		req.Header.Set("Host", u.Host)
	}
}

// SetToken 外部设置 Token（chromedp 方式获取后调用）
func (cm *CameraManager) SetToken(token string) {
	cm.tokenMu.Lock()
	cm.token = token
	cm.tokenMu.Unlock()
	cm.markLoginReady()
	log.Println("[摄像头] Token 已外部设置并就绪")
}

// Token 获取当前 Token
func (cm *CameraManager) Token() string {
	cm.tokenMu.RLock()
	defer cm.tokenMu.RUnlock()
	return cm.token
}

// HasToken 检查是否有 Token
func (cm *CameraManager) HasToken() bool {
	cm.tokenMu.RLock()
	defer cm.tokenMu.RUnlock()
	return cm.token != ""
}
