@echo off
setlocal EnableDelayedExpansion

set "ROOT=C:\Developer\soccer\footy-model"
cd /d "%ROOT%"
if not exist "artifacts\logs" mkdir "artifacts\logs"
if not exist "artifacts\locks" mkdir "artifacts\locks"
set "LOCKFILE=artifacts\locks\odds_polling_scheduler.lock"
set "STALE_MINUTES=360"

if exist "%LOCKFILE%" (
  powershell -NoProfile -Command "$p='%LOCKFILE%'; $age=((Get-Date).ToUniversalTime() - (Get-Item $p).LastWriteTimeUtc).TotalMinutes; if ($age -ge %STALE_MINUTES%) { Remove-Item $p -Force; exit 2 } else { exit 1 }" >nul 2>&1
  set "LOCK_CHECK_RC=!ERRORLEVEL!"
  if "!LOCK_CHECK_RC!"=="2" (
    echo [%DATE% %TIME%] Removed stale odds lock older than %STALE_MINUTES% minutes.>> artifacts\logs\odds_polling_scheduler.stdout.log
  ) else (
    echo [%DATE% %TIME%] Skip: odds polling scheduler already running.>> artifacts\logs\odds_polling_scheduler.stdout.log
    exit /b 0
  )
)

type nul > "%LOCKFILE%"

"C:\Python313\python.exe" src\jobs\sofascore_odds_polling.py --days 3 --sleep-sec 0.5 --log-file artifacts\logs\odds_polling_scheduler.log >> artifacts\logs\odds_polling_scheduler.stdout.log 2>&1
set "RC=%ERRORLEVEL%"

del "%LOCKFILE%" >nul 2>&1

endlocal
exit /b %RC%
