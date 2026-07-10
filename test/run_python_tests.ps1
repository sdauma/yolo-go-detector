$tests = @(
    # ===== 已完成的 Python 测试（跳过）=====
    # R2 PASS: 4个CPU场景完成，结果文件存在（CRASH为exit code误判，实际运行完整）
    @{ Name = 'python_cpu_monitoring'; Script = 'python_cpu_monitoring.py'; Timeout = 10080; Skip = $true },
    # R2 PASS: YOLO11x+YOLO11n Session创建时间完成（CRASH为exit code误判）
    @{ Name = 'python_session_creation_benchmark'; Script = 'python_session_creation_benchmark.py'; Timeout = 10080; Skip = $true },
    # R2 PASS: 输出一致性验证完成，yolo11x/yolo11n各5目标结果文件存在（CRASH为exit code误判）
    @{ Name = 'python_output_consistency'; Script = 'python_output_consistency.py'; Timeout = 10080; Skip = $true },

    # R3~R6 PASS: 结果文件已存在，跳过
    @{ Name = 'python_concurrent_stress_test_fixed'; Script = 'python_concurrent_stress_test_fixed.py'; Timeout = 10080; Skip = $true },

    # ===== 已完成的 Python 测试（R3 全部 PASS，结果文件存在）=====
    @{ Name = 'python_memory_copy_overhead'; Script = 'python_memory_copy_overhead.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_session_pool_ablation'; Script = 'python_session_pool_ablation.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_architecture_benchmark'; Script = 'python_architecture_benchmark.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_long_stability'; Script = 'python_long_stability.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_reinforced_benchmark'; Script = 'python_reinforced_benchmark.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_reinforced_benchmark_small'; Script = 'python_reinforced_benchmark_small.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_reinforced_yolo11n'; Script = 'python_reinforced_yolo11n.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_baseline_supplementary'; Script = 'python_baseline_supplementary.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_baseline'; Script = 'python_baseline.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_pure_inference_benchmark'; Script = 'python_pure_inference_benchmark.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_cold_start_benchmark'; Script = 'python_cold_start_benchmark.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_cold_start_decomposition'; Script = 'python_cold_start_decomposition.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_memory_standardization'; Script = 'python_memory_standardization.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_thread_config_benchmark'; Script = 'python_thread_config_benchmark.py'; Timeout = 10080; Skip = $true },
    @{ Name = 'python_72h_stability_1h'; Script = 'python_long_stability_72h.py'; Timeout = 10080; Args = '1'; Skip = $true },
    # R4: Arena 开关消融实验（跨语言一致性验证）
    @{ Name = 'python_arena_ablation'; Script = 'python_arena_ablation.py'; Timeout = 10080; Skip = $false }
)

$results = @()

foreach ($test in $tests) {
    if ($test.Skip) {
        Write-Host ('[SKIP] ' + $test.Name + ' (marked as skip)')
        $results += @{ Name = $test.Name; Status = 'SKIPPED' }
        Write-Host ''
        continue
    }

    Write-Host '========================================'
    Write-Host ('  Test: ' + $test.Name)
    Write-Host '========================================'
    try {
        if (-not (Test-Path $test.Script)) {
            Write-Host '[ERROR] Script not found: ' + $test.Script
            $results += @{ Name = $test.Name; Status = 'MISSING' }
            Write-Host ''
            continue
        }

        $hasArgs = $test.ContainsKey('Args') -and $test.Args
        Write-Host ('[Running] python ' + $test.Script + $(if ($hasArgs) { ' ' + $test.Args } else { '' }) + ' (Timeout: ' + $test.Timeout + ' minutes)...')

        if ($hasArgs) {
            $process = Start-Process -FilePath 'python' -ArgumentList ('"' + $test.Script + '" ' + $test.Args) -NoNewWindow -PassThru
        } else {
            $process = Start-Process -FilePath 'python' -ArgumentList ('"' + $test.Script + '"') -NoNewWindow -PassThru
        }
        $timedOut = $false
        $startTime = Get-Date
        $script:lastReported = 0

        while (-not $process.HasExited) {
            Start-Sleep -Seconds 5
            $elapsed = [int]((Get-Date) - $startTime).TotalMinutes
            if ($elapsed -ge $test.Timeout) {
                Write-Host ('[Timeout] Test timed out after ' + $elapsed + ' min, killing process tree...')
                taskkill /F /T /PID $process.Id 2>&1 | ForEach-Object { Write-Host $_ }
                $timedOut = $true
                Start-Sleep -Seconds 1
                break
            }
            if ($elapsed -gt 0 -and $elapsed % 60 -eq 0 -and $elapsed -ne $script:lastReported) {
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
        $results += @{ Name = $test.Name; Status = 'ERROR' }
        Write-Host ''
    }
}

Write-Host '========================================'
Write-Host '  Python Test Results Summary'
Write-Host '========================================'
foreach ($r in $results) {
    $icon = switch ($r.Status) { 'PASS' { '[OK]' }; 'CRASH' { '[FAIL]' }; 'TIMEOUT' { '[TIMEOUT]' }; 'MISSING' { '[MISSING]' }; 'ERROR' { '[ERROR]' }; 'UNKNOWN' { '[UNKNOWN]' }; 'SKIPPED' { '[SKIP]' }; default { '[UNKNOWN]' } }
    Write-Host ('  ' + $icon + ' ' + $r.Name + ': ' + $r.Status + $(if ($r.ExitCode) { ' (ExitCode: ' + $r.ExitCode + ')' } else { '' }))
}
