param(
    [string]$CondaExecutable = "D:\Anaconda3\Scripts\conda.exe",
    [string]$CondaEnvironment = "py37"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryScript = Join-Path $projectRoot "torch_hierarchicalfl_mnist_lr_step_by_step_example.py"
$matRelativePath = "matlab/result-U-6fixedge_epoch200.mat"
$matPath = Join-Path $projectRoot $matRelativePath
$experiments = @(
    @{
        Name = "HFL+SnF"
        Config = "configs/fedml_config_hfl_snf_fixed_u05.yaml"
    },
    @{
        Name = "HFL-noSnF"
        Config = "configs/fedml_config_hfl_no_snf_fixed_u05.yaml"
    },
    @{
        Name = "FL+SnF"
        Config = "configs/fedml_config_fl_snf_u05.yaml"
    },
    @{
        Name = "FL-noSnF"
        Config = "configs/fedml_config_fl_no_snf_u05.yaml"
    }
)

# Validate all shared inputs before starting the first long-running experiment.
if (-not (Test-Path -LiteralPath $CondaExecutable -PathType Leaf)) {
    throw "Conda executable does not exist: $CondaExecutable"
}
if (-not (Test-Path -LiteralPath $entryScript -PathType Leaf)) {
    throw "MNIST experiment entry does not exist: $entryScript"
}
if (-not (Test-Path -LiteralPath $matPath -PathType Leaf)) {
    throw "MAT file does not exist: $matPath"
}

$expectedCandidateLine = "fixed_candidate_client_ids: [123, 124, 125, 126, 127, 128, 41, 42, 43, 44, 45, 46, 0, 1, 2, 3, 4, 5, 164, 165, 166, 167, 168, 169, 82, 83, 84, 85, 86, 87, 205, 206, 207, 208, 209, 210, 129]"
foreach ($experiment in $experiments) {
    $configPath = Join-Path $projectRoot $experiment.Config
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Experiment config does not exist: $configPath"
    }
    $configText = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8
    if ($configText -notmatch "(?m)^\s*client_num_in_total:\s*250\s*$") {
        throw "client_num_in_total must be 250: $configPath"
    }
    if ($configText -notmatch "(?m)^\s*client_num_per_round:\s*37\s*$") {
        throw "client_num_per_round must be 37: $configPath"
    }
    if (-not $configText.Contains($expectedCandidateLine)) {
        throw "Fixed candidate client list is inconsistent: $configPath"
    }
    if ($configText -notmatch '(?m)^\s*topology_assignment_mode:\s*"balanced_counts"\s*(?:#.*)?$') {
        throw "topology_assignment_mode must be balanced_counts: $configPath"
    }
    if (-not $configText.Contains($matRelativePath)) {
        throw "Experiment config does not reference the required MAT file: $configPath"
    }
}

# Validate the named Conda environment without starting training.
& $CondaExecutable run -n $CondaEnvironment python --version
if ($LASTEXITCODE -ne 0) {
    throw "Conda environment is unavailable: $CondaEnvironment"
}

Write-Host "MNIST 250-client four-experiment suite is starting." -ForegroundColor Cyan
Write-Host "Project root : $projectRoot"
Write-Host "Conda env    : $CondaEnvironment"
Write-Host "MAT file     : $matPath"
Write-Host "Run order    : $($experiments.Name -join ', ')"
Write-Host "The four experiments run serially and print logs in real time."

$suiteStartTime = Get-Date
Push-Location $projectRoot
try {
    for ($index = 0; $index -lt $experiments.Count; $index++) {
        $experiment = $experiments[$index]
        $configPath = Join-Path $projectRoot $experiment.Config
        $experimentStartTime = Get-Date
        Write-Host ""
        Write-Host (
            "[{0}/{1}] Starting {2}: {3}" -f `
                ($index + 1), $experiments.Count, $experiment.Name, $experiment.Config
        ) -ForegroundColor Yellow

        # --no-capture-output keeps FedML round logs visible in the current terminal.
        & $CondaExecutable run --no-capture-output -n $CondaEnvironment `
            python $entryScript --cf $configPath
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($experiment.Name) failed with exit code $LASTEXITCODE."
        }

        $experimentElapsed = (Get-Date) - $experimentStartTime
        Write-Host (
            "[{0}/{1}] Completed {2} in {3:dd\.hh\:mm\:ss}." -f `
                ($index + 1), $experiments.Count, $experiment.Name, $experimentElapsed
        ) -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

$suiteElapsed = (Get-Date) - $suiteStartTime
Write-Host ""
Write-Host (
    "All four MNIST experiments completed in {0:dd\.hh\:mm\:ss}." -f $suiteElapsed
) -ForegroundColor Green
