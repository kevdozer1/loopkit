# Start the full bake-off as plain Python. No agent involved.
#
#   powershell -ExecutionPolicy Bypass -File setup\start_bakeoff.ps1
#   powershell -ExecutionPolicy Bypass -File setup\start_bakeoff.ps1 -Date 2026-10-18   # resume that run
#
# Loads .env through env.ps1, checks the 10 GB disk floor on C: and D:, prints the cost estimate,
# then runs
#   uv run python -m loopkit.bakeoff.run --config config/bakeoff.yaml --date <date>
# with stdout written to LOOPKIT_DATA\bakeoff\<date>\run.log (stderr to run.err.log), and finally
# builds the report (report.md, pass_vs_cost.png, results.parquet, a copy in reports\).
# Every unit is checkpointed, so the same command with the same -Date resumes where it stopped.
# Models without a key in .env are skipped and listed in the log and the report.

param([string]$Date = (Get-Date -Format 'yyyy-MM-dd'))

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo 'env.ps1')
$env:PYTHONUNBUFFERED = '1'

foreach ($drive in 'C', 'D') {
    $d = Get-PSDrive -Name $drive -ErrorAction SilentlyContinue
    if ($null -ne $d -and ($d.Free / 1GB) -lt 10) {
        Write-Host ("Stopped: {0}: has {1:N1} GB free, under the 10 GB floor." -f $drive, ($d.Free / 1GB))
        exit 1
    }
}

$out = Join-Path $env:LOOPKIT_DATA (Join-Path 'bakeoff' $Date)
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'run.log'
$err = Join-Path $out 'run.err.log'

Write-Host 'Cost estimate for config\bakeoff.yaml:'
& uv run python -m loopkit.bakeoff.estimate --config config/bakeoff.yaml

Write-Host "Bake-off $Date starting. Progress: Get-Content -Wait '$log'"
$run = Start-Process -FilePath 'uv' -NoNewWindow -Wait -PassThru `
    -ArgumentList @('run', 'python', '-m', 'loopkit.bakeoff.run', '--config', 'config/bakeoff.yaml',
                    '--date', $Date, '--log', '-', '--no-report') `
    -RedirectStandardOutput $log -RedirectStandardError $err
Write-Host "Runner exit code: $($run.ExitCode)"

& uv run python -m loopkit.bakeoff.report --date $Date
Write-Host "Report: $out\report.md (copied to reports\bakeoff_$Date.md)"
exit $run.ExitCode
