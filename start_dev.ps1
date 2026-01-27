# Start Backend and Frontend in separate windows
Write-Host "Starting full development environment..."

Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\start_backend.ps1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\start_frontend.ps1"

Write-Host "Development environment launched in separate windows!"
