param(
    [string]$CondaExecutable = "D:\Anaconda3\Scripts\conda.exe",
    [string]$CondaEnvironment = "py37",
    [int]$GpuId = 0,
    [int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$matRelativePath = "matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
$matPath = Join-Path $projectRoot $matRelativePath
$configNames = @(
    "femnist_hfl_snf_u05_5000.yaml",
    "femnist_hfl_no_snf_u05_5000.yaml",
    "femnist_fl_snf_u05_5000.yaml",
    "femnist_fl_no_snf_u05_5000.yaml"
)

# Validate the interpreter, MAT file, and all formal configs before training.
if (-not (Test-Path -LiteralPath $CondaExecutable -PathType Leaf)) {
    throw "Conda executable does not exist: $CondaExecutable"
}
if (-not (Test-Path -LiteralPath $matPath -PathType Leaf)) {
    throw "MAT file does not exist: $matPath"
}
if ($BatchSize -le 0) {
    throw "BatchSize must be a positive integer."
}
foreach ($configName in $configNames) {
    $configPath = Join-Path $projectRoot "FEMNISTProbe\configs\$configName"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Experiment config does not exist: $configPath"
    }
    $configText = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8
    if (-not $configText.Contains($matRelativePath)) {
        throw "Experiment config does not reference the required MAT file: $configPath"
    }
}

Write-Host "FEMNIST four-experiment suite is starting." -ForegroundColor Cyan
Write-Host "Project root : $projectRoot"
Write-Host "Conda env    : $CondaEnvironment"
Write-Host "GPU id       : $GpuId"
Write-Host "Batch size   : $BatchSize"
Write-Host "MAT file     : $matPath"
Write-Host "Run order    : $($configNames -join ', ')"
Write-Host "Every communication round will be printed and copied to suite job logs."

$startTime = Get-Date
Push-Location $projectRoot
try {
    # Serial mode keeps round logs ordered while run_suite retains suite files.
    & $CondaExecutable run --no-capture-output -n $CondaEnvironment `
        python -m FEMNISTProbe.run_suite `
        --mode formal `
        --gpu_id $GpuId `
        --parallel 1 `
        --batch_size $BatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "FEMNIST four-experiment suite failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$elapsed = (Get-Date) - $startTime
Write-Host (
    "All four FEMNIST experiments completed in {0:dd\.hh\:mm\:ss}." -f $elapsed
) -ForegroundColor Green
