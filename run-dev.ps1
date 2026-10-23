# Run from any working directory. Reload applies Python changes automatically.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --workers 1 --reload --reload-dir "$PSScriptRoot\rag"
