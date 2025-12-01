@echo off
REM Example helper to enable the workspace for using Google Generative AI (Codex-like models).
REM Edit the API key below or set the environment variable globally.

echo This script will run list_models.py with either an env var or a passed API key.

if "%1"=="" (
    echo Usage: enable_codex.bat <YOUR_GOOGLE_API_KEY>
    echo OR set environment variables using 'setx GOOGLE_API_KEY <KEY>' and re-open the shell.
    echo.
    echo Running without arguments will try the environment variable GOOGLE_API_KEY
    python ..\list_models.py
) else (
    set GOOGLE_API_KEY=%1
    python ..\list_models.py --api-key %1
)

pause
