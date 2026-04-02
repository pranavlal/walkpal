<#
    WalkingPal Windows 10/11 One-Click Setup
    Handles Python detection, VENV isolation, and Desktop Shortcut creation.
#>

$AppName = "WalkingPal"
$InstallDir = $PSScriptRoot
$PythonMinVersion = "3.10"

Write-Host ">>> Starting $AppName Windows Setup <<<" -ForegroundColor Green

# 1. Check for Python
try {
    $pythonVersion = & python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $ver = $matches[1]
        Write-Host "Found Python $ver"
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "Python not detected. Please install Python $PythonMinVersion+ from python.org" -ForegroundColor Red
    Pause
    exit
}

# 2. Setup Virtual Environment
Write-Host "Initializing Virtual Environment..."
if (-not (Test-Path "$InstallDir\.venv")) {
    & python -m venv .venv
}

$PipCmd = "$InstallDir\.venv\Scripts\pip.exe"
$PythonExe = "$InstallDir\.venv\Scripts\python.exe"
$PythonWExe = "$InstallDir\.venv\Scripts\pythonw.exe"

if (-not (Test-Path "$InstallDir\requirements.txt")) {
    Write-Host "CRITICAL Error: requirements.txt not found in $InstallDir" -ForegroundColor Red
    exit 1
}

# 3. Install Dependencies
Write-Host "Installing Dependencies (this may take a few minutes)..."
& $PipCmd install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Pip upgrade failed" }
& $PipCmd install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Requirements installation failed" }

# 4. Create Desktop Shortcut
Write-Host "Creating Desktop Shortcut..."
$Shell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.IO.Path]::Combine([System.Environment]::GetFolderPath("Desktop"), "$AppName.lnk")
$Shortcut = $Shell.CreateShortcut($DesktopPath)
$Shortcut.TargetPath = "$PythonExe"
$Shortcut.Arguments = "$InstallDir\walkingPal.py"
$Shortcut.WorkingDirectory = "$InstallDir"
$Shortcut.Description = "Walking Assistant for the Blind"
$Shortcut.Save()

Write-Host ">>> Setup Complete! You can now run $AppName from your Desktop. <<<" -ForegroundColor Green
Pause
