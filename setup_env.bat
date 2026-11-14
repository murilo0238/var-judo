@echo off
REM Cria o ambiente virtual e instala as dependencias do VAR Judo.
cd /d "%~dp0"

set PYEXE=
where py >nul 2>nul && (
    py -3.12 --version >nul 2>nul && set PYEXE=py -3.12
)
if "%PYEXE%"=="" (
    REM Sem Python 3.12: usa o Python padrao disponivel.
    py --version >nul 2>nul && set PYEXE=py
)
if "%PYEXE%"=="" set PYEXE=python

echo Usando interpretador: %PYEXE%
%PYEXE% -m venv .venv
if errorlevel 1 (
    echo Falha ao criar o ambiente virtual.
    pause
    exit /b 1
)

call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo Baixando modelo de pose (se necessario)...
call ".venv\Scripts\python.exe" -c "import os,urllib.request as u; p='var_judo/models/pose_landmarker_lite.task'; os.makedirs('var_judo/models',exist_ok=True); (not os.path.exists(p)) and u.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', p); print('modelo OK')"

echo Validando imports...
call ".venv\Scripts\python.exe" -c "import mediapipe,cv2,PyQt5,openai,numpy,PIL; print('Ambiente OK')"

echo.
echo Pronto! Defina OPENAI_API_KEY e rode run.bat
pause
