<#
.SYNOPSIS
    run_experiments.ps1
    Runs a list of Main.py experiments sequentially, logs each to its own file,
    and continues to the next experiment even if one fails.

.USAGE
    .\run_experiments.ps1

    Optional: restrict to one GPU
    $env:CUDA_VISIBLE_DEVICES = "0"; .\run_experiments.ps1
#>

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
$PythonBin  = "python"          # use "python" on Windows (not python3)
$MainScript = "Main.py"
$LogDir     = "exp_logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Each entry is one full set of CLI args for Main.py.
# Add / remove / edit experiments here.
$Experiments = @(
    "--type diffusion --algo diff --model_prefix TransformerAdaSECosineDiffusionv0 --mode eval --finetuning_ckpt exp/TransformerAdaSECosineDiffusionv0/models/model_ckpt.pth.tar --exp_desc ode_mask --enable_complete True --deterministic True",
    "--type diffusion --algo diff --model_prefix TransformerAdaSEFlatCosineDiffusionv0 --mode eval --finetuning_ckpt exp/TransformerAdaSEFlatCosineDiffusionv0/models/model_ckpt.pth.tar --exp_desc ode_mask --enable_complete True --deterministic True",
    "--type diffusion --algo diff --model_prefix TransformerStySECosineDiffusionv0 --mode eval --finetuning_ckpt exp/TransformerStySECosineDiffusionv0/models/model_ckpt.pth.tar --exp_desc ode_mask --enable_complete True --deterministic True",
    "--type diffusion --algo diff --model_prefix TransformerStySEFlatCosineDiffusionv0 --mode eval --finetuning_ckpt exp/TransformerStySEFlatCosineDiffusionv0/models/model_ckpt.pth.tar --exp_desc ode_mask --enable_complete True --deterministic True"
    # add more experiments here...
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
$Total  = $Experiments.Count
$Failed = @()

Write-Host "Running $Total experiment(s). Logs will be written to $LogDir/"
Write-Host "----------------------------------------------------------------"

for ($i = 0; $i -lt $Total; $i++) {
    $ArgsLine = $Experiments[$i]
    $Idx      = $i + 1
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile  = Join-Path $LogDir "exp_${Idx}_${Timestamp}.log"

    Write-Host ""
    Write-Host "[$Idx/$Total] Starting: $ArgsLine"
    Write-Host "  -> logging to $LogFile"

    $StartTime = Get-Date

    # Split the args string on whitespace into an array.
    # (Fine as long as no individual arg value itself contains a space;
    # none of the experiments above do.)
    $ArgArray = $ArgsLine -split '\s+' | Where-Object { $_ -ne '' }

    # Run and tee output to both console and log file, capturing exit code
    & $PythonBin $MainScript @ArgArray 2>&1 | Tee-Object -FilePath $LogFile
    $Status = $LASTEXITCODE

    $Elapsed = [int]((Get-Date) - $StartTime).TotalSeconds

    if ($Status -eq 0) {
        Write-Host "[$Idx/$Total] DONE (${Elapsed}s): $ArgsLine"
    } else {
        Write-Host "[$Idx/$Total] FAILED (exit $Status, ${Elapsed}s): $ArgsLine"
        $Failed += "[$Idx] $ArgsLine"
    }
}

Write-Host ""
Write-Host "----------------------------------------------------------------"
Write-Host "All $Total experiment(s) finished."

if ($Failed.Count -eq 0) {
    Write-Host "All experiments succeeded."
    exit 0
} else {
    Write-Host "$($Failed.Count) experiment(s) failed:"
    foreach ($f in $Failed) {
        Write-Host "  - $f"
    }
    exit 1
}