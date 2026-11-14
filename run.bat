@echo off
REM Inicia o VAR Judo usando o ambiente virtual do projeto.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente virtual nao encontrado. Rode setup_env.bat primeiro.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m var_judo.main
if errorlevel 1 pause
