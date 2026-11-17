# DP-SGD privacy-utility sweep (2026-11-16): two looser privacy budgets than eps = 8,
# same recipe otherwise (800 steps x 512 expected examples, C = 1, delta = 1/(10N)).
# Each run is followed by its attacks; figures are regenerated at the end.
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$log = "checkpoints\queue_dp_sweep.log"

function Py($argsList) {
    "=== $(Get-Date -Format s) python $($argsList -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList 2>&1 | Out-File -Append -Encoding utf8 $log
    "exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}

foreach ($eps in @(64, 32)) {
    $run = "dp_eps${eps}_M"
    $dpArgs = @("src\dp_sgd.py", "--run", $run, "--model", "M", "--train_bin", "data\tokens_canary\train.bin",
                "--epsilon", "$eps", "--max_steps", "800", "--batch_size", "512")
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
