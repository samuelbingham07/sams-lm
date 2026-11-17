# After the DP sweep: rerun attacks on the four earlier models so their canary
# results include the beam-search extraction attack, then regenerate figures.
param([int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$log = "checkpoints\queue_beam_rerun.log"
if ($WaitPid -gt 0) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }

function Py($argsList) {
    "=== $(Get-Date -Format s) python $($argsList -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList 2>&1 | Out-File -Append -Encoding utf8 $log
    "exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}

foreach ($run in @("baseline_M", "canary_M", "canary_dedup_M", "dp_eps8_M")) {
    $csv = "experiments\results\$run\attacks\canaries.csv"
    if (-not ((Test-Path $csv) -and ((Get-Content $csv -TotalCount 1) -match "extracted_beam")) -or -not (Test-Path "experiments\results\$run\attacks\summary.json")) {
        Py @("src\extract.py", "--run", $run)
    }
}
Py @("src\plots.py")
"DONE" | Out-File -Append -Encoding utf8 $log
