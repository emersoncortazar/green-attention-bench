$env:PYTHONPATH = (Resolve-Path "..\src").Path
Write-Host "PYTHONPATH set to $env:PYTHONPATH"