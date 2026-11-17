# Extras (2026-11-17): shadow-model MIA, large-batch DP-SGD, memorization over training.
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$log = "checkpoints\queue_extras.log"

function Py($argsList) {
    "=== $(Get-Date -Format s) python $($argsList -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList 2>&1 | Out-File -Append -Encoding utf8 $log
    "exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}

# 1. Shadow model: the target's recipe (model M, 1 epoch) on the shadow half, different seed.
if (-not (Test-Path "checkpoints\shadow_M\final.pt")) {
    $a = @("src\train.py", "--run", "shadow_M", "--model", "M", "--train_bin", "data\tokens_shadow\train.bin", "--seed", "7")
    if (Test-Path "checkpoints\shadow_M\ckpt.pt") { $a += "--resume" }
    Py $a
}

# 2. DP-SGD with 8x the batch (4096 expected examples), eps = 8, 200 steps (~420M tokens).
#    Noise on the averaged gradient shrinks like sigma / batch, so this should cut the utility cost.
$run = "dp_bigbatch_eps8_M"
if (-not (Test-Path "checkpoints\$run\final.pt")) {
    $a = @("src\dp_sgd.py", "--run", $run, "--model", "M", "--train_bin", "data\tokens_canary\train.bin",
           "--epsilon", "8", "--max_steps", "200", "--batch_size", "4096", "--warmup_steps", "20", "--eval_every", "50")
    if (Test-Path "checkpoints\$run\ckpt.pt") { $a += "--resume" }
    Py $a
}
if ((Test-Path "checkpoints\$run\final.pt") -and -not (Test-Path "experiments\results\$run\attacks\summary.json")) {
    Py @("src\extract.py", "--run", $run)
}

# 3. Memorization over training: attack the canary_M snapshots.
foreach ($s in @(2500, 5000, 7500, 10000, 12500)) {
    if (-not (Test-Path "experiments\results\canary_M\attacks_step$s\summary.json")) {
        Py @("src\extract.py", "--run", "canary_M", "--ckpt", "checkpoints\canary_M\model_step$s.pt", "--tag", "step$s")
    }
}

# 4. Shadow-model attacks on every target.
if (Test-Path "checkpoints\shadow_M\final.pt") { Py @("src\shadow.py", "attack") }

Py @("src\plots.py")
"DONE" | Out-File -Append -Encoding utf8 $log
