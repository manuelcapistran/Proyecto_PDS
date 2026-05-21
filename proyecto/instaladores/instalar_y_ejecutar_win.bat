@echo off
echo ============================================
echo   Grabador de Audio - PDS
echo ============================================
echo.
echo Instalando dependencias (solo la primera vez)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias.
    echo Asegurate de tener Python instalado correctamente.
    echo Descargalo en: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.
echo Iniciando programa...
python grabacion_gui.py
pause
