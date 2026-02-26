@echo off
setlocal EnableDelayedExpansion

set "ROOT=C:\Developer\soccer\footy-model"
cd /d "%ROOT%"
if not exist "artifacts\logs" mkdir "artifacts\logs"
if not exist "artifacts\locks" mkdir "artifacts\locks"
set "LOCKFILE=artifacts\locks\availability_scheduler.lock"
set "STALE_MINUTES=360"

if exist "%LOCKFILE%" (
  powershell -NoProfile -Command "$p='%LOCKFILE%'; $age=((Get-Date).ToUniversalTime() - (Get-Item $p).LastWriteTimeUtc).TotalMinutes; if ($age -ge %STALE_MINUTES%) { Remove-Item $p -Force; exit 2 } else { exit 1 }" >nul 2>&1
  set "LOCK_CHECK_RC=!ERRORLEVEL!"
  if "!LOCK_CHECK_RC!"=="2" (
    echo [%DATE% %TIME%] Removed stale availability lock older than %STALE_MINUTES% minutes.>> artifacts\logs\availability_scheduler.stdout.log
  ) else (
    echo [%DATE% %TIME%] Skip: availability scheduler already running.>> artifacts\logs\availability_scheduler.stdout.log
    exit /b 0
  )
)

type nul > "%LOCKFILE%"

"C:\Python313\python.exe" src\ingest\ingest_sofascore_availability.py --status scheduled --limit 250 --scheduled-start-hours 0 --scheduled-end-hours 72 --scheduled-order asc --retry-404-cooldown-hours 6 >> artifacts\logs\availability_scheduler.stdout.log 2>&1
set "RC=%ERRORLEVEL%"

del "%LOCKFILE%" >nul 2>&1

endlocal
exit /b %RC%
