# Poketrader App

Welcome to the Poketrader App project! This application consists of a Python backend (FastAPI) and a React Native frontend (Expo).

## Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (LTS version recommended)
- [Python](https://www.python.org/) (3.10 or later)
- [Expo Go](https://expo.dev/go) app on your mobile device (iOS or Android)
- PowerShell (for running the start scripts on Windows)

## Project Structure

- `app/backend`: Python FastAPI backend
- `app/mobile`: React Native Expo frontend
- `start_backend.ps1`: Script to start the backend server
- `start_frontend.ps1`: Script to start the Expo development server
- `start_dev.ps1`: Script to start both backend and frontend

## Setup

### 1. Backend Setup

The backend handles the business logic and AI processing.

1.  Navigate to the backend directory:
    ```powershell
    cd app/backend
    ```

2.  Create a virtual environment (if not executing the scripts automatically):
    ```powershell
    python -m venv venv
    ```

3.  Activate the virtual environment:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```

4.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

### 2. Frontend Setup

The frontend is a mobile application built with React Native and Expo.

1.  Navigate to the mobile directory:
    ```powershell
    cd app/mobile
    ```

2.  Install dependencies:
    ```powershell
    npm install
    ```

## Running the Application

### Option 1: Using Start Scripts (Recommended)

You can run the backend and frontend separately using the provided PowerShell scripts in the root directory.

- **Start Backend**:
  ```powershell
  .\start_backend.ps1
  ```
  This will start the API server on `http://0.0.0.0:8001`.

- **Start Frontend**:
  ```powershell
  .\start_frontend.ps1
  ```
  This will start the Expo development server. Scan the QR code with the Expo Go app to run it on your device.

### Option 2: Manual Start

**Backend**:
```powershell
cd app/backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

**Frontend**:
```powershell
cd app/mobile
npx expo start
```

## Troubleshooting

- **Backend Connection**: Ensure your mobile device is on the same Wi-Fi network as your computer. You may need to update the API URL in the frontend configuration if it's hardcoded to `localhost` (though `start_backend` uses `0.0.0.0`, your phone needs your computer's local IP).
- **Expo Issues**: If you encounter issues with Expo, try clearing the cache with `npx expo start -c`.
