@echo off
setlocal EnableDelayedExpansion

set "ROOT=C:\Developer\soccer\footy-model"
cd /d "%ROOT%"
if not exist "artifacts\logs" mkdir "artifacts\logs"
if not exist "artifacts\locks" mkdir "artifacts\locks"
if not exist "artifacts\reports" mkdir "artifacts\reports"
if not exist "artifacts\reports\layer2_feature_health" mkdir "artifacts\reports\layer2_feature_health"
if not exist "artifacts\reports\monitoring" mkdir "artifacts\reports\monitoring"
if not exist "artifacts\reports\leakage_audit" mkdir "artifacts\reports\leakage_audit"

set "LOCKFILE=artifacts\locks\validation_loop_scheduler.lock"
set "STALE_MINUTES=360"
set "STDOUT_LOG=artifacts\logs\validation_loop_scheduler.stdout.log"
set "LEAKAGE_SOURCE=.sisyphus\evidence\task-7-leakage-audit.json"

if exist "%LOCKFILE%" (
  powershell -NoProfile -Command "$p='%LOCKFILE%'; $age=((Get-Date).ToUniversalTime() - (Get-Item $p).LastWriteTimeUtc).TotalMinutes; if ($age -ge %STALE_MINUTES%) { Remove-Item $p -Force; exit 2 } else { exit 1 }" >nul 2>&1
  set "LOCK_CHECK_RC=!ERRORLEVEL!"
  if "!LOCK_CHECK_RC!"=="2" (
    echo [%DATE% %TIME%] Removed stale validation lock older than %STALE_MINUTES% minutes.>> "%STDOUT_LOG%"
  ) else (
    echo [%DATE% %TIME%] Skip: validation loop already running.>> "%STDOUT_LOG%"
    exit /b 0
  )
)

type nul > "%LOCKFILE%"

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString(\"yyyyMMdd_HHmmss\")"') do set "RUN_TS=%%I"
if not defined RUN_TS set "RUN_TS=unknown_run"

set "HEALTH_OUT=artifacts\reports\layer2_feature_health\%RUN_TS%"
set "MONITOR_OUT=artifacts\reports\monitoring\%RUN_TS%"
set "LEAKAGE_COPY=artifacts\reports\leakage_audit\leakage_audit_%RUN_TS%.json"
set "RC=0"

echo [%DATE% %TIME%] Validation loop start run_ts=%RUN_TS%.>> "%STDOUT_LOG%"

echo COMMAND: C:\Python313\python.exe src\modeling\layer2_situational\audit_feature_health.py --output-dir "%HEALTH_OUT%">> "%STDOUT_LOG%"
"C:\Python313\python.exe" src\modeling\layer2_situational\audit_feature_health.py --output-dir "%HEALTH_OUT%" >> "%STDOUT_LOG%" 2>&1
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" set "RC=!STEP_RC!"

echo COMMAND: C:\Python313\python.exe src\db\audit_layer2_situational_leakage.py --days 14 --limit 500>> "%STDOUT_LOG%"
"C:\Python313\python.exe" src\db\audit_layer2_situational_leakage.py --days 14 --limit 500 >> "%STDOUT_LOG%" 2>&1
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" set "RC=!STEP_RC!"
if exist "%LEAKAGE_SOURCE%" copy /Y "%LEAKAGE_SOURCE%" "%LEAKAGE_COPY%" >nul

echo COMMAND: C:\Python313\python.exe src\modeling\layer2_situational\monitor_layer2_signals.py --days 14 --output-dir "%MONITOR_OUT%">> "%STDOUT_LOG%"
"C:\Python313\python.exe" src\modeling\layer2_situational\monitor_layer2_signals.py --days 14 --output-dir "%MONITOR_OUT%" >> "%STDOUT_LOG%" 2>&1
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" set "RC=!STEP_RC!"

echo [%DATE% %TIME%] Validation loop end run_ts=%RUN_TS% rc=!RC!.>> "%STDOUT_LOG%"

del "%LOCKFILE%" >nul 2>&1

endlocal
exit /b %RC%
