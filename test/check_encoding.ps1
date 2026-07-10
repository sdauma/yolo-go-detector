$files = Get-ChildItem "d:\mlz\trae_projects\1\yolo-go-detector\test\python\*.py"
foreach ($file in $files) {
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $hasNonAscii = $false
    foreach ($b in $bytes) {
        if ($b -gt 127) {
            $hasNonAscii = $true
            break
        }
    }
    if ($hasNonAscii) {
        Write-Host "NON-ASCII: $($file.Name)"
    } else {
        Write-Host "OK: $($file.Name)"
    }
}