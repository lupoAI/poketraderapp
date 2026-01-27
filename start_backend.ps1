# Start Backend (Port 8001 with Auto-Reload)
Write-Host "Starting Backend on Port 8001..."
PUSH-LOCATION app/backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
POP-LOCATION
