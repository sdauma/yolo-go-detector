Write-Host "=== 长时间稳定性测试内存监控脚本 ===" -ForegroundColor Green

# 配置参数
$warmupRuns = 10
$totalRuns = 300
$sampleInterval = 10
$midPoint = $totalRuns / 2

# 存储数据
$rssData = @()
$startRSS = 0
$midRSS = 0
$endRSS = 0

Write-Host "测试配置:"
Write-Host "- Warmup runs: $warmupRuns"
Write-Host "- Total runs: $totalRuns"
Write-Host "- Sample interval: $sampleInterval runs"
Write-Host "- Mid point: $midPoint runs"
Write-Host ""

# 启动 Go 稳定性测试
Write-Host "启动 Go 长时间稳定性测试..." -ForegroundColor Cyan
$goProcess = Start-Process -FilePath "go" -ArgumentList "run", "benchmark_go_long_stability.go" -PassThru

Write-Host "Go 进程 ID: $($goProcess.Id)" -ForegroundColor Yellow
Write-Host "开始监控内存使用情况..." -ForegroundColor Cyan
Write-Host ""

# 等待 warmup 完成
Write-Host "等待 warmup 阶段完成..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# 记录起始 RSS
Write-Host "" 
Write-Host "=== Warmup 完成，记录起始 RSS ===" -ForegroundColor Green
$process = Get-Process -Id $goProcess.Id -ErrorAction SilentlyContinue
if ($process) {
    $startRSS = [math]::Round($process.WS / 1MB, 1)
    $rssData += [PSCustomObject]@{Iteration=0; RSS_MB=$startRSS}
    Write-Host "Start RSS: $startRSS MB" -ForegroundColor Green
}

# 监控主循环
$currentRun = 0
$lastSample = 0

Write-Host "" 
Write-Host "=== 开始监控主测试阶段 ===" -ForegroundColor Cyan

while ($currentRun -lt $totalRuns) {
    # 检查进程是否仍在运行
    $process = Get-Process -Id $goProcess.Id -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Host "Go 进程已结束，停止监控" -ForegroundColor Red
        break
    }

    # 每 10 次运行记录一次 RSS
    if ($currentRun -ge $lastSample + $sampleInterval) {
        $currentRSS = [math]::Round($process.WS / 1MB, 1)
        $rssData += [PSCustomObject]@{Iteration=$currentRun; RSS_MB=$currentRSS}
        Write-Host "Run $currentRun, RSS: $currentRSS MB" -ForegroundColor Cyan
        $lastSample = $currentRun

        # 记录中间点 RSS
        if ($currentRun -eq $midPoint) {
            $midRSS = $currentRSS
            Write-Host "" 
            Write-Host "=== 记录中间点 RSS ===" -ForegroundColor Yellow
            Write-Host "Mid RSS: $midRSS MB" -ForegroundColor Yellow
        }
    }

    # 模拟运行计数（实际运行次数由 Go 程序控制）
    $currentRun += 1
    Start-Sleep -Milliseconds 500
}

# 记录结束 RSS
Write-Host "" 
Write-Host "=== 测试完成，记录结束 RSS ===" -ForegroundColor Green
$process = Get-Process -Id $goProcess.Id -ErrorAction SilentlyContinue
if ($process) {
    $endRSS = [math]::Round($process.WS / 1MB, 1)
    $rssData += [PSCustomObject]@{Iteration=$totalRuns; RSS_MB=$endRSS}
    Write-Host "End RSS: $endRSS MB" -ForegroundColor Green
}

# 计算漂移
$drift = [math]::Round((max (abs ($startRSS - $midRSS)) (abs ($midRSS - $endRSS))), 1)

Write-Host "" 
Write-Host "=== 长时间稳定性测试结果 ===" -ForegroundColor Green
Write-Host "Start RSS: $startRSS MB"
Write-Host "Mid RSS: $midRSS MB"
Write-Host "End RSS: $endRSS MB"
Write-Host "Drift: ±$drift MB"
Write-Host "" 

# 保存结果到文件
$results = @"
===== 长时间稳定性测试结果 =====
Start RSS: $startRSS MB
Mid RSS: $midRSS MB
End RSS: $endRSS MB
Drift: ±$drift MB
"@

$results | Out-File -FilePath "stability_results.txt" -Force
Write-Host "结果已保存到 stability_results.txt" -ForegroundColor Cyan

# 保存详细数据
$rssData | Export-Csv -Path "rss_details.csv" -NoTypeInformation
Write-Host "详细 RSS 数据已保存到 rss_details.csv" -ForegroundColor Cyan

Write-Host "" 
Write-Host "=== 监控完成 ===" -ForegroundColor Green

# 辅助函数
function max($a, $b) {
    if ($a -gt $b) { return $a }
    return $b
}

function abs($x) {
    if ($x -lt 0) { return -$x }
    return $x
}
