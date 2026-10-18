$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$runName = 'runs/demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $projectPython -m adaptive_defense.experiment --demo --output $runName --epochs 20 --episodes 100 --window 64
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. See the output above.' }
