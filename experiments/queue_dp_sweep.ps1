# DP-SGD privacy-utility sweep (2026-11-16). Same recipe as dp_eps8_M
# (800 steps x 512 expected examples, C = 1, delta = 1/(10N)); each run is
# followed by its attacks, figures are regenerated at the end.
#
# History: the first version asked for eps = 64 and 32, but the noise search had
# a floor at sigma = 0.3, so both got sigma = 0.30 (eps = 29.4). The first run is
# kept as dp_eps29_M; the duplicate was deleted; the search now fails loudly.
#
# Runs here:
#   dp_sigma0_M  control: the DP-SGD recipe with no noise (eps = infinity). Separates
#                "noise protects the canaries" from "this recipe just learns less".
#   dp_eps64_M   a looser budget (sigma solved correctly this time).
param([int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$log = "checkpoints\queue_dp_sweep.log"
if ($WaitPid -gt 0) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }

function Py($argsList) {
    "=== $(Get-Date -Format s) python $($argsList -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList 2>&1 | Out-File -Append -Encoding utf8 $log
    "exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}

$runs = [ordered]@{
    "dp_sigma0_M" = @("--noise_multiplier", "0")
    "dp_eps64_M"  = @("--epsilon", "64")
}
foreach ($run in $runs.Keys) {
    $dpArgs = @("src\dp_sgd.py", "--run", $run, "--model", "M", "--train_bin", "data\tokens_canary\train.bin",
                "--max_steps", "800", "--batch_size", "512") + $runs[$run]
    if (-not (Test-Path "checkpoints\$run\final.pt")) {
        if (Test-Path "checkpoints\$run\ckpt.pt") { $dpArgs += "--resume" }
        Py $dpArgs
    }
    if ((Test-Path "checkpoints\$run\final.pt") -and -not (Test-Path "experiments\results\$run\attacks\summary.json")) {
        Py @("src\extract.py", "--run", $run)
    }
}
Py @("src\plots.py")
"DONE" | Out-File -Append -Encoding utf8 $log
