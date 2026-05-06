@echo off
setlocal

set "ROOT=%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%ROOT%scripts\easy_dataset_workflow.py" %*
  set "EXIT_CODE=%errorlevel%"
  if not "%EXIT_CODE%"=="0" pause
  exit /b %EXIT_CODE%
)

where python >nul 2>nul
if %errorlevel%==0 (
  python "%ROOT%scripts\easy_dataset_workflow.py" %*
  set "EXIT_CODE=%errorlevel%"
  if not "%EXIT_CODE%"=="0" pause
  exit /b %EXIT_CODE%
)

echo Python wurde nicht gefunden.
echo Bitte installiere Python 3 und fuehre danach die Datei erneut aus.
pause
exit /b 1
