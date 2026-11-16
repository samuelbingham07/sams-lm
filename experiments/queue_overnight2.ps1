# Second overnight queue (2026-11-16): runs after queue_overnight.ps1 finishes.
# Attacks on every model, then the DP-SGD run, then figures.
param([int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$log = "checkpoints\queue2.log"
if ($WaitPid -gt 0) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }

function Py($argsList) {
    "=== $(Get-Date -Format s) python $($argsList -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList 2>&1 | Out-File -Append -Encoding utf8 $log
}

foreach ($run in @("baseline_M", "canary_M", "canary_dedup_M")) {
    if ((Test-Path "checkpoints\$run\final.pt") -and -not (Test-Path "experiments\results\$run\attacks\summary.json")) {
        Py @("src\extract.py", "--run", $run)
    }
}

# Defense 2: DP-SGD at epsilon = 8 on the canary data.
$dpArgs = @("src\dp_sgd.py", "--run", "dp_eps8_M", "--model", "M", "--train_bin", "data\tokens_canary\train.bin",
            "--epsilon", "8", "--max_steps", "800", "--batch_size", "512")
if (-not (Test-Path "checkpoints\dp_eps8_M\final.pt")) {
    if (Test-Path "checkpoints\dp_eps8_M\ckpt.pt") { $dpArgs += "--resume" }
    Py $dpArgs
}
if ((Test-Path "checkpoints\dp_eps8_M\final.pt") -and -not (Test-Path "experiments\results\dp_eps8_M\attacks\summary.json")) {
    Py @("src\extract.py", "--run", "dp_eps8_M")
}
Py @("src\plots.py")
