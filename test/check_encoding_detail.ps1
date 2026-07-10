$files = @(
    "python_architecture_benchmark.py",
    "python_baseline.py",
    "python_baseline_supplementary.py",
    "python_concurrent_stress_test_fixed.py",
    "python_cpu_monitoring.py",
    "python_memory_copy_overhead.py",
    "python_output_consistency.py",
    "python_session_creation_benchmark.py",
    "python_session_pool_ablation.py"
)

$basePath = "d:\mlz\trae_projects\1\yolo-go-detector\test\python"

foreach ($fileName in $files) {
    $filePath = Join-Path $basePath $fileName
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    $lines = [System.Text.Encoding]::UTF8.GetString($bytes).Split("`n")
    
    Write-Host "`n=== $fileName ==="
    for ($i = 0; $i -lt $lines.Length; $i++) {
        $line = $lines[$i]
        for ($j = 0; $j -lt $line.Length; $j++) {
            $c = $line[$j]
            if ([int]$c -gt 127) {
                Write-Host "Line $($i+1), Col $($j+1): '$c' (Unicode: U+$([int]$c).ToString('X4'))"
            }
        }
    }
}