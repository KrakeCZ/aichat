@echo off
setlocal
cd /d "%~dp0"
where pyw.exe >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys, tkinter; sys.exit(sys.version_info < (3,10))" >nul 2>&1
    if not errorlevel 1 (
        start "" pyw.exe -3 "%~dp0AdbHardwareMonitor.pyw"
        exit /b 0
    )
)
where pythonw.exe >nul 2>&1
if not errorlevel 1 (
    python -c "import sys, tkinter; sys.exit(sys.version_info < (3,10))" >nul 2>&1
    if not errorlevel 1 (
        start "" pythonw.exe "%~dp0AdbHardwareMonitor.pyw"
        exit /b 0
    )
)
echo Nainstaluj Python 3.10 nebo novejsi s podporou Tcl/Tk.
echo https://www.python.org/downloads/windows/
pause
exit /b 1
