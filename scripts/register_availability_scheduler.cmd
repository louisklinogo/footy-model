@echo off
setlocal

set "ROOT=C:\Developer\soccer\footy-model"
set "RUNNER=%ROOT%\scripts\run_availability_scheduler.cmd"
set "TASK_NAME=FootyAvailabilityScheduler"

if not exist "%RUNNER%" (
  echo Missing runner script: %RUNNER%
  exit /b 1
)

schtasks /Create /TN "%TASK_NAME%" /SC HOURLY /MO 1 /TR "%RUNNER%" /F
if errorlevel 1 exit /b %ERRORLEVEL%

schtasks /Query /TN "%TASK_NAME%" /V /FO LIST
endlocal
exit /b %ERRORLEVEL%
