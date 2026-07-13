param(
    [string]$CondaExecutable = "D:\Anaconda3\Scripts\conda.exe",
    [string]$CondaEnvironment = "py37"
)

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryScript = Join-Path $scriptDirectory "torch_hierarchicalfl_mnist_lr_step_by_step_example.py"
$configFiles = @(
    "configs/fedml_config_hfl_snf_fixed_u05.yaml",
    "configs/fedml_config_hfl_no_snf_fixed_u05.yaml",
    "configs/fedml_config_fl_snf_u05.yaml",
    "configs/fedml_config_fl_no_snf_u05.yaml"
)
#
# if (-not (Test-Path -LiteralPath $CondaExecutable)) {
#     throw "找不到 Conda：$CondaExecutable"
# }
#
Push-Location $scriptDirectory
try {
    foreach ($configFile in $configFiles) {
        Write-Host "Start :$configFile"
        # --no-capture-output 让 FedML 日志实时显示在当前终端中。
        & $CondaExecutable run --no-capture-output -n $CondaEnvironment `
            python $entryScript --cf $configFile
        if ($LASTEXITCODE -ne 0) {
            throw "Fail :$configFile，Exit code :$LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
