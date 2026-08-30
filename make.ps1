# Windows shim for the Makefile targets — `make` is not installed by default on
# Windows. Same targets, same commands. Usage: .\make.ps1 validate
param([Parameter(Position = 0)][string]$Target = "validate")

$ErrorActionPreference = "Stop"
$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Invoke-Step($cmd, $argv) {
    Write-Host "> $cmd $($argv -join ' ')" -ForegroundColor Cyan
    & $cmd @argv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "install" {
        Invoke-Step $PY @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Step $PY @("-m", "pip", "install", "-r", "requirements.txt")
    }
    "test" { Invoke-Step $PY @("-m", "pytest", "tests/", "-q") }
    "lint" {
        Invoke-Step $PY @("-m", "ruff", "check", "src/", "app/", "eval/", "tests/")
        Invoke-Step $PY @("-m", "ruff", "format", "--check", "src/", "app/", "eval/", "tests/")
    }
    "validate" {
        & $PSCommandPath lint
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath test
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "validate OK" -ForegroundColor Green
    }
    "run" { Invoke-Step $PY @("-m", "streamlit", "run", "app/streamlit_app.py") }
    "eval" { Invoke-Step $PY @("eval/run_eval.py") }
    default {
        Write-Host "Unknown target '$Target'."
        Write-Host "Targets: install, test, lint, validate, run, eval"
        exit 1
    }
}
