@echo off
setlocal

if "%~1"=="--from-temp" (
    set "REPO_ROOT=%~2"
    goto main
)

pushd "%~dp0" >nul
set "REPO_ROOT=%CD%"
popd >nul
set "TEMP_BAT=%TEMP%\update_git_%RANDOM%_%RANDOM%.bat"
copy /y "%~f0" "%TEMP_BAT%" >nul
call "%TEMP_BAT%" --from-temp "%REPO_ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"
del "%TEMP_BAT%" >nul 2>&1
exit /b %EXIT_CODE%

:main
cd /d "%REPO_ROOT%"

echo [1/4] Checking repository...
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This folder is not a Git repository.
    pause
    exit /b 1
)

for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH=%%b"
if "%BRANCH%"=="" (
    echo Could not detect the current branch.
    pause
    exit /b 1
)

echo Current branch: %BRANCH%

echo [2/4] Checking for local changes...
for /f "delims=" %%s in ('git status --porcelain') do goto dirty

echo [3/4] Fetching origin...
git fetch origin
if errorlevel 1 (
    echo Fetch failed.
    pause
    exit /b 1
)

echo [4/4] Pulling latest changes...
git pull --ff-only origin %BRANCH%
if errorlevel 1 (
    echo Pull failed. The branch may need a manual merge or rebase.
    pause
    exit /b 1
)

echo.
echo Git update complete.
pause
exit /b 0

:dirty
echo.
echo Local changes were detected. Commit or stash them before updating.
echo.
git status --short
echo.
choice /c SN /m "Stash local changes, update Git, then re-apply them? [S=stash/update, N=cancel]"
if errorlevel 2 (
    echo Update canceled.
    pause
    exit /b 1
)

echo.
echo Stashing local changes...
git stash push -u -m "auto-stash before update_git.bat"
if errorlevel 1 (
    echo Stash failed.
    pause
    exit /b 1
)

echo Fetching origin...
git fetch origin
if errorlevel 1 (
    echo Fetch failed. Your changes are saved in git stash.
    echo Restoring stashed changes...
    git stash pop
    pause
    exit /b 1
)

echo Pulling latest changes...
git pull --ff-only origin %BRANCH%
if errorlevel 1 (
    echo Pull failed. Your changes are saved in git stash.
    echo Restoring stashed changes...
    git stash pop
    pause
    exit /b 1
)

echo Re-applying stashed changes...
git stash pop
if errorlevel 1 (
    echo Stash pop had conflicts or failed. Resolve conflicts manually.
    pause
    exit /b 1
)

echo.
echo Git update complete. Local changes were re-applied.
pause
exit /b 0
