$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot '.venv\python.exe') -m litter_detector.ui
exit $LASTEXITCODE
