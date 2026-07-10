$logFile = $args[0]

$tests = @(
    # ===== 全部24个Go测试程序已完成（第1轮16个 + 第2轮8个），第三次运行全部跳过 =====
    # R1 PASS: 10轮全部完成
    @{ Name = 'go_baseline_minimal'; Exe = 'go_baseline_minimal.exe'; Source = 'go_baseline_minimal.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 10轮全部完成
    @{ Name = 'go_reinforced_benchmark'; Exe = 'go_reinforced_benchmark.exe'; Source = 'go_reinforced_benchmark.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 10轮全部完成
    @{ Name = 'go_reinforced_yolo11n'; Exe = 'go_reinforced_yolo11n.exe'; Source = 'go_reinforced_yolo11n.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 10轮全部完成
    @{ Name = 'go_reinforced_benchmark_small'; Exe = 'go_reinforced_benchmark_small.exe'; Source = 'go_reinforced_benchmark_small.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: YOLO11x + YOLO11n 完成
    @{ Name = 'go_pure_inference_benchmark'; Exe = 'go_pure_inference_benchmark.exe'; Source = 'go_pure_inference_benchmark.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 5次独立测试完成
    @{ Name = 'cold_start_benchmark'; Exe = 'cold_start_benchmark.exe'; Source = 'cold_start_benchmark.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 内存分解完成
    @{ Name = 'go_memory_breakdown'; Exe = 'go_memory_breakdown.exe'; Source = 'go_memory_breakdown.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: Session创建时间完成
    @{ Name = 'go_session_creation_benchmark'; Exe = 'go_session_creation_benchmark.exe'; Source = 'go_session_creation_benchmark.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 输出一致性验证完成
    @{ Name = 'go_output_consistency'; Exe = 'go_output_consistency.exe'; Source = 'go_output_consistency.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 4个CPU监控场景全部完成
    @{ Name = 'go_cpu_monitoring'; Exe = 'go_cpu_monitoring.exe'; Source = 'go_cpu_monitoring.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 预热效应分析完成
    @{ Name = 'go_warmup_effect'; Exe = 'go_warmup_effect.exe'; Source = 'go_warmup_effect.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: S-A1~S-A4 完成
    @{ Name = 'go_advanced_session_supplementary'; Exe = 'go_advanced_session_supplementary.exe'; Source = 'go_advanced_session_supplementary.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: 200次推理诊断完成
    @{ Name = 'go_performance_diagnostic'; Exe = 'go_performance_diagnostic.exe'; Source = 'go_performance_diagnostic.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: YOLO11x 10次 + YOLO11n 10次完成
    @{ Name = 'go_memory_standardization'; Exe = 'go_memory_standardization.exe'; Source = 'go_memory_standardization.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: YOLO11x 20次 + YOLO11n 20次完成
    @{ Name = 'go_cold_start_decomposition'; Exe = 'go_cold_start_decomposition.exe'; Source = 'go_cold_start_decomposition.go'; Timeout = 10080; Skip = $true },
    # R1 PASS: Batch 1/4/8/16/32 完成
    @{ Name = 'go_batch_inference'; Exe = 'go_batch_inference.exe'; Source = 'go_batch_inference.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 全部6个实验完成（架构对比+消融+冷启动+稳定性+批处理），exit code=-1为误判
    @{ Name = 'paper_full_benchmark'; Exe = 'paper_full_benchmark.exe'; Source = 'paper_full_benchmark.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 32组消融实验完成（30组OK+2组SKIP）
    @{ Name = 'go_session_pool_ablation'; Exe = 'go_session_pool_ablation.exe'; Source = 'go_session_pool_ablation.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 1~12并发全部完成
    @{ Name = 'go_concurrent_stress_fixed'; Exe = 'go_concurrent_stress_fixed.exe'; Source = 'go_concurrent_stress_fixed.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 内存拷贝开销+线程调度全部完成
    @{ Name = 'go_memory_copy_overhead'; Exe = 'go_memory_copy_overhead.exe'; Source = 'go_memory_copy_overhead.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 4种线程配置全部完成（intra=1/2/4/8各5次）
    @{ Name = 'thread_config_benchmark'; Exe = 'thread_config_benchmark.exe'; Source = 'thread_config_benchmark.go'; Timeout = 10080; Skip = $true },
    # R3: 重新运行（修复 wg.Wait() Bug，原数据中 Session Pool 吞吐量/RSS 全部错误）
    @{ Name = 'go_architecture_benchmark'; Exe = 'go_architecture_benchmark.exe'; Source = 'go_architecture_benchmark.go'; Timeout = 10080; Skip = $false },
    # R2 PASS: 1~12并发三架构对比全部完成
    @{ Name = 'go_concurrent_architecture_comparison'; Exe = 'go_concurrent_architecture_comparison.exe'; Source = 'go_concurrent_architecture_comparison.go'; Timeout = 10080; Skip = $true },
    # R2 PASS: 600次推理完成，PM漂移3.45MB
    @{ Name = 'go_72h_stability_1h'; Exe = 'go_72h_stability.exe'; Source = 'go_72h_stability.go'; Timeout = 10080; Args = '1'; Skip = $true },
    # R4: Arena 开关消融实验（验证 Unsafe Shared 漂移来自共享 Arena）
    @{ Name = 'go_arena_ablation'; Exe = 'go_arena_ablation.exe'; Source = 'go_arena_ablation.go'; Timeout = 10080; Skip = $false }
)

$results = @()

foreach ($test in $tests) {
    if ($test.Skip) {
        Write-Host ('[SKIP] ' + $test.Name + ' (marked as skip)')
        Add-Content -Path $logFile -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] [SKIP] ' + $test.Name)
        $results += @{ Name = $test.Name; Status = 'SKIPPED' }
        Write-Host ''
        continue
    }

    Write-Host '========================================'
    Write-Host ('  Test: ' + $test.Name)
    Write-Host '========================================'
    Add-Content -Path $logFile -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] [GO] ' + $test.Name)

    try {
        # Auto-build if exe missing
        if (-not (Test-Path $test.Exe)) {
            Write-Host ('[BUILD] ' + $test.Exe + ' not found, auto-building...')
            Add-Content -Path $logFile -Value ('[BUILD] ' + $test.Exe + ' not found, auto-building from ' + $test.Source)
            $buildResult = go build -o $test.Exe $test.Source 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host ('[FAIL] Auto-build failed for ' + $test.Source)
                Write-Host ('  ' + ($buildResult -join "`n  "))
                Add-Content -Path $logFile -Value ('[BUILD_FAIL] ' + $test.Source + ': ' + ($buildResult -join '; '))
                $results += @{ Name = $test.Name; Status = 'BUILD_FAIL' }
                Write-Host ''
                continue
            }
            Write-Host ('[OK] Auto-build succeeded: ' + $test.Exe)
            Add-Content -Path $logFile -Value ('[BUILD_OK] ' + $test.Exe)
        }

        Write-Host ('[Running] ' + $test.Exe + ' (Timeout: ' + $test.Timeout + ' minutes)...')
        $hasArgs = $test.ContainsKey('Args') -and $test.Args
        Add-Content -Path $logFile -Value ('[RUN] ' + $test.Exe + $(if ($hasArgs) { ' ' + $test.Args } else { '' }))

        if ($hasArgs) {
            $process = Start-Process -FilePath ('.\' + $test.Exe) -ArgumentList $test.Args -NoNewWindow -PassThru
        } else {
            $process = Start-Process -FilePath ('.\' + $test.Exe) -NoNewWindow -PassThru
        }
        $timedOut = $false
        $startTime = Get-Date
        $script:lastReported = 0

        while (-not $process.HasExited) {
            Start-Sleep -Seconds 5
            $elapsed = [int]((Get-Date) - $startTime).TotalMinutes
            if ($elapsed -ge $test.Timeout) {
                Write-Host ('[Timeout] Test timed out after ' + $elapsed + ' min, killing process tree...')
                Add-Content -Path $logFile -Value ('[TIMEOUT] Killing process tree after ' + $elapsed + ' min')
                taskkill /F /T /PID $process.Id 2>&1 | ForEach-Object { Write-Host $_; Add-Content -Path $logFile -Value $_ }
                $timedOut = $true
                Start-Sleep -Seconds 1
                break
            }
            if ($elapsed -gt 0 -and $elapsed % 60 -eq 0 -and $elapsed -ne $lastReported) {
                Write-Host ('[Still running] ' + $test.Name + ' - ' + $elapsed + ' min elapsed, please wait...')
                $script:lastReported = $elapsed
            }
        }

        if ($timedOut) {
            $results += @{ Name = $test.Name; Status = 'TIMEOUT' }
        } else {
            try {
                if ($process.HasExited) {
                    $exitCode = $process.ExitCode
                    Add-Content -Path $logFile -Value ('[EXITCODE] ' + $(if ($exitCode -eq $null) { 'null (clean exit)' } else { $exitCode }))
                    if ($exitCode -ne $null -and $exitCode -ne 0) {
                        $results += @{ Name = $test.Name; Status = 'CRASH'; ExitCode = $exitCode }
                    } else {
                        $results += @{ Name = $test.Name; Status = 'PASS' }
                    }
                } else {
                    $results += @{ Name = $test.Name; Status = 'UNKNOWN' }
                }
            } catch {
                $results += @{ Name = $test.Name; Status = 'CRASH'; ExitCode = -1 }
                Write-Host ('[Exception] Failed to read exit code: ' + $_.Exception.Message)
            }
        }

        try { $process.Close(); $process.Dispose() } catch {}

        Write-Host ''
    } catch {
        Write-Host ('[Exception] ' + $_.Exception.Message)
        Add-Content -Path $logFile -Value ('[ERROR] Exception: ' + $_.Exception.Message)
        $results += @{ Name = $test.Name; Status = 'ERROR' }
        Write-Host ''
    }
}

Write-Host '========================================'
Write-Host '  Go Test Results Summary'
Write-Host '========================================'
foreach ($r in $results) {
    $icon = switch ($r.Status) { 'PASS' { '[OK]' }; 'CRASH' { '[FAIL]' }; 'TIMEOUT' { '[TIMEOUT]' }; 'MISSING_EXE' { '[NO EXE]' }; 'ERROR' { '[ERROR]' }; 'UNKNOWN' { '[UNKNOWN]' }; 'SKIPPED' { '[SKIP]' }; default { '[UNKNOWN]' } }
    Write-Host ('  ' + $icon + ' ' + $r.Name + ': ' + $r.Status + $(if ($r.ExitCode) { ' (ExitCode: ' + $r.ExitCode + ')' } else { '' }))
}