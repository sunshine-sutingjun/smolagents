#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Gaia 运行脚本
.DESCRIPTION
    检查 Python 环境和虚拟环境后，执行 run_gaia.py
#>
[CmdletBinding()]
param(
    [Alias('c')]
    [int]    $Concurrency     = 1,

    [Alias('m')]
    [string] $ModelId         = 'Qwen/Qwen2.5-72B-Instruct-128K',

    [Alias('r')]
    [Parameter(Mandatory=$false)]
    [string] $RunName='test-run',

    [Alias('s')]
    [string] $SetToRun        = 'test',

    [Alias('o')]
    [switch] $UseOpenModels,

    [Alias('d')]
    [switch] $UseRawDataset,

    [Alias('h')]
    [switch] $Help
)

# 检查Python环境
$pythonPath = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonPath) {
    Write-Error "Python not found. Please install Python and add it to your PATH."
    exit 1
}

# 检查虚拟环境存在
$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Error "Virtual environment not found at '$venvPath'. Please create it first:"  
    Write-Host "  python -m venv $venvPath"  
    Write-Host "  .\$venvPath\Scripts\Activate"  
    Write-Host "  pip install -e ."  
    exit 1
}

# 构建命令行参数列表
$args = @(
    ".\examples\open_deep_research\run_gaia.py"
    "--concurrency $Concurrency"
    "--model-id `"$ModelId`""
    "--run-name `"$RunName`""
    "--set-to-run `"$SetToRun`""
)

if ($UseOpenModels) { $args += "--use-open-models" }
if ($UseRawDataset) { $args += "--use-raw-dataset" }

# 显示并执行
$cmdLine = "python " + ($args -join " ")
Write-Host "Executing: $cmdLine"
Start-Process -FilePath "python" -ArgumentList $args -NoNewWindow -Wait
