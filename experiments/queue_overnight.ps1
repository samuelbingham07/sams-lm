# Overnight training queue (2026-11-16). Runs each job after the previous one exits.
# Launch detached:  Start-Process powershell -ArgumentList "-File","experiments\queue_overnight.ps1" -WindowStyle Hidden
param([int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

if ($WaitPid -gt 0) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }

function Run($name, $argsList) {
    if (Test-Path "checkpoints\$name\final.pt") { return }   # already finished
    $a = @("-u", "src\train.py", "--run", $name) + $argsList
    $ckpt = "checkpoints\$name\ckpt.pt"
    if (Test-Path $ckpt) { $a += "--resume" }
    New-Item -ItemType Directory -Force "checkpoints" | Out-Null
    & $py @a 1>> "checkpoints\${name}_stdout.log" 2>> "checkpoints\${name}_stderr.log"
}

# 1. Canaries planted, no defense (same recipe and seed as baseline_M).
Run "canary_M" @("--model", "M", "--train_bin", "data\tokens_canary\train.bin")
# 2. Defense 1: exact dedup. Same number of steps as canary_M (compute-matched).
Run "canary_dedup_M" @("--model", "M", "--train_bin", "data\tokens_canary_dedup\train.bin", "--max_steps", "14778")
