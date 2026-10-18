# Run loopkit's kickoff or one session headless, with no permission prompts.
#
#   powershell -ExecutionPolicy Bypass -File setup\run_session.ps1            # KICKOFF (S0 to S4)
#   powershell -ExecutionPolicy Bypass -File setup\run_session.ps1 -Session S2
#
# Needs Claude Code 2.1.259 or later and the autoMode settings from setup\claude_settings_snippet.json
# merged into C:\Users\Kevin\.claude\settings.json.
# Afterwards: read reports\<Session>.md, then search reports\<Session>_run.json for permission_denials.
# To pick the session back up: claude --resume <session_id from that json>

param([string]$Session = 'KICKOFF')

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo 'env.ps1')

if ($Session -eq 'KICKOFF') {
    $prompt = 'Read KICKOFF.md and follow it exactly. Keep reports/PROGRESS.md current. When done or stopped, end with the message format in KICKOFF.md section 4.'
} else {
    $prompt = "Read CLAUDE.md, then follow sessions/$Session.md exactly. When done or stopped, write your report to reports/$Session.md and update reports/PROGRESS.md."
}

$out = Join-Path $repo "reports\${Session}_run.json"
Write-Host "Running $Session headless. Output: $out"
$result = claude -p $prompt --permission-mode auto --permission-prompts none --output-format json
[IO.File]::WriteAllText($out, ($result -join "`n"), [Text.UTF8Encoding]::new($false))
Write-Host "Done. Read reports\$Session.md (or reports\PROGRESS.md for KICKOFF)."
