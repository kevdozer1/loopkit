# Dot-source this to load .env into the current PowerShell session:  . .\env.ps1
# CRLF safe. Same pattern as statebench\env.ps1.
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { return }
        $i = $line.IndexOf('=')
        if ($i -lt 1) { return }
        $k = $line.Substring(0, $i).Trim()
        $v = $line.Substring($i + 1).Trim()
        if ($v -ne '') { Set-Item -Path ("env:" + $k) -Value $v }
    }
} else {
    Write-Host "No .env found. Copy .env.example to .env and fill in keys."
}
if (-not $env:LOOPKIT_DATA) { $env:LOOPKIT_DATA = 'D:\loopkit_data' }
$env:UV_LINK_MODE = 'copy'
if ($env:MUJOCO_GL -eq 'egl') { Remove-Item Env:MUJOCO_GL }
$env:Path = "C:\Users\Kevin\.local\bin;$env:Path"
