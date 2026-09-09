@echo off

setlocal

cd /d "%~dp0"

rem Dev-checkout wrapper. The portable build ships its own copy of this that uses

rem the bundled Python instead of the venv.

if not exist "%~dp0venv\Scripts\python.exe" (

    echo venv not found. Create it and pip install -r requirements.txt first.

    exit /b 1

)

"%~dp0venv\Scripts\python.exe" "%~dp0src\switch_model.py" %*

