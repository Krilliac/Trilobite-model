@echo off
setlocal
set "REPO=%~dp0"
"%REPO%venv\Scripts\python.exe" "%REPO%trilobite_client.py" %*
endlocal
