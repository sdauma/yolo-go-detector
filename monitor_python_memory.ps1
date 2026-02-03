Write-Host "=== Python 内存监控脚本 ===" -ForegroundColor Green

$maxRSS = 0
$rssValues = @()
$startTime = Get-Date

Write-Host "开始监控 Python 进程内存使用..." -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止监控" -ForegroundColor Yellow

try {
    while ($true) {
        $processes = Get-Process -Name "python*" -ErrorAction SilentlyContinue
        
        foreach ($process in $processes) {
            $currentWS = $process.WS
            $currentWS_MB = [math]::Round($currentWS / 1MB, 2)
            
            # 更新最大 RSS
            if ($currentWS -gt $maxRSS) {
                $maxRSS = $currentWS
                $maxRSS_MB = [math]::Round($maxRSS / 1MB, 2)
                Write-Host "[峰值] Python 进程ID: $($process.Id), RSS: $maxRSS_MB MB" -ForegroundColor Red
            }
            
            # 记录 RSS 值
            $rssValues += $currentWS_MB
            if ($rssValues.Count -gt 50) {
                $rssValues = $rssValues[-50..-1]
            }
            
            $elapsed = (Get-Date) - $startTime
            Write-Host "[实时] 运行时间: $($elapsed.ToString('mm\:ss')), 进程ID: $($process.Id), RSS: $currentWS_MB MB" -ForegroundColor Cyan
        }
        
        if ($processes.Count -eq 0) {
            Write-Host "[状态] 未找到 Python 进程" -ForegroundColor Gray
        }
        
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "监控中断" -ForegroundColor Yellow
}

Write-Host "\n=== 监控结果 ===" -ForegroundColor Green
Write-Host "最大 RSS: $([math]::Round($maxRSS / 1MB, 2)) MB" -ForegroundColor Red

if ($rssValues.Count -gt 0) {
    # 计算稳定期 RSS（取最后 10 个值的平均值）
    $stableCount = [Math]::Min(10, $rssValues.Count)
    $stableRSS = ($rssValues[-$stableCount..-1] | Measure-Object -Average).Average
    Write-Host "稳定 RSS: $([math]::Round($stableRSS, 2)) MB" -ForegroundColor Green
    Write-Host "记录值数量: $($rssValues.Count)" -ForegroundColor Cyan
}

Write-Host "=== 监控结束 ===" -ForegroundColor Green
