$ErrorActionPreference = "Stop"

# Script is at HFLSnF_KG/run_fixed_fedtranse_comparisons.ps1
# $PSScriptRoot = HFLSnF_KG/, project root = parent of that
$projectRoot = Split-Path -Parent $PSScriptRoot

$configFiles = @(
    (Join-Path $projectRoot "HFLSnF_KG/configs/server_fb15k237_flnosnf_fixed_cuda.yaml"),
    (Join-Path $projectRoot "HFLSnF_KG/configs/server_fb15k237_flsnf_fixed_cuda.yaml"),
    (Join-Path $projectRoot "HFLSnF_KG/configs/server_fb15k237_hflnosnf_fixed_cuda.yaml"),
    (Join-Path $projectRoot "HFLSnF_KG/configs/server_fb15k237_hflsnf_fixed_cuda.yaml")
)

Push-Location -LiteralPath $projectRoot
try {
    $env:PYTHONPATH = "$projectRoot"
    foreach ($configFile in $configFiles) {
        Write-Host "========================================"
        Write-Host "Running fixed federated TransE: $configFile"
        Write-Host "========================================"
        python -m HFLSnF_KG.run_fixed_federated_transe --cf $configFile
        if ($LASTEXITCODE -ne 0) {
            throw "Config failed: $configFile (exit code: $LASTEXITCODE)"
        }
    }
    Write-Host "========================================"
    Write-Host "All 4 fixed federated TransE experiments completed!"
    Write-Host "========================================"
}
finally {
    Pop-Location
}