param([string]$CondaPython = 'D:\miniconda3\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:CONDA_NO_PLUGINS = 'true'
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.cache\conda-pkgs'
$env:CONDA_ENVS_PATH = Join-Path $projectRoot '.cache\conda-envs'
if (-not (Test-Path -LiteralPath '.venv\python.exe')) {
    & $CondaPython -m conda env create --prefix '.\.venv' --file environment.yml --solver classic
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
}
& '.\.venv\python.exe' -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& '.\.venv\python.exe' -c 'import torch; print(torch.__version__); print("CUDA available:",torch.cuda.is_available())'
