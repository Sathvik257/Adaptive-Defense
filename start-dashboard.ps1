$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the virtual environment using the README setup commands first.'
}
& $projectPython -m streamlit run dashboard.py --server.address 127.0.0.1
