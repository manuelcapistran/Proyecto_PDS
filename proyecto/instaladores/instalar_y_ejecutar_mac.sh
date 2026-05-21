#!/bin/bash
echo "============================================"
echo "  Grabador de Audio - PDS"
echo "============================================"
echo ""
echo "Instalando dependencias (solo la primera vez)..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: No se pudieron instalar las dependencias."
    echo "Asegurate de tener Python instalado correctamente."
    echo "Descargalo en: https://www.python.org/downloads/"
    read -p "Presiona Enter para cerrar..."
    exit 1
fi
echo ""
echo "Iniciando programa..."
python3 grabacion_gui.py
