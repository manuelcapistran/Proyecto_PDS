# 🎙️ Sintetizador de Voz con LPC

## 📋 Descripción
Este programa implementa un sintetizador de voz completo usando Codificación Predictiva Lineal (LPC) que incluye:

- ✅ Grabación de audio desde micrófono
- ✅ Guardado en formato WAV
- ✅ Pre-énfasis de la señal
- ✅ Ventanas de Hamming
- ✅ Cálculo de autocorrelación
- ✅ Algoritmo de Levinson-Durbin
- ✅ Síntesis LPC
- ✅ Visualización de espectrogramas (original vs sintetizado)

## 🔧 Instalación de Dependencias

```bash
pip install numpy scipy sounddevice matplotlib
```

### Dependencias del sistema (Linux/macOS)
Si tienes problemas con el audio, instala:

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
Las librerías generalmente funcionan sin configuración adicional.

## 🚀 Uso

Simplemente ejecuta:

```bash
python sintetizador_voz.py
```

El programa automáticamente:
1. Te pedirá que hables durante 3 segundos
2. Guardará el audio original como `audio_original.wav`
3. Procesará el audio usando LPC
4. Guardará el audio sintetizado como `audio_sintetizado.wav`
5. Generará espectrogramas comparativos en `espectrogramas.png`
6. Reproducirá ambos audios para comparación

## 📊 Archivos Generados

- `audio_original.wav` - Tu grabación original
- `audio_sintetizado.wav` - Audio reconstruido con LPC
- `espectrogramas.png` - Gráficos de comparación

## 🔬 Parámetros Técnicos

Puedes ajustar estos parámetros en el código:

```python
sintetizador = SintetizadorVoz(
    duracion=3,        # Segundos de grabación
    fs=16000,          # Frecuencia de muestreo (Hz)
    orden_lpc=12       # Orden del filtro LPC
)
```

### Orden LPC recomendado:
- **Voz masculina:** 10-12
- **Voz femenina:** 12-14
- **Mayor calidad:** 14-16 (más procesamiento)

## 🎯 Proceso Técnico

### 1. Pre-énfasis
Enfatiza frecuencias altas: `y[n] = x[n] - α·x[n-1]` (α = 0.97)

### 2. Ventana de Hamming
Reduce efectos de borde: `w[n] = 0.54 - 0.46·cos(2πn/(N-1))`

### 3. Autocorrelación
Calcula correlación de la señal consigo misma usando FFT

### 4. Levinson-Durbin
Resuelve ecuaciones de Yule-Walker eficientemente:
- Calcula coeficientes del filtro LPC
- Minimiza el error de predicción

### 5. Síntesis
Genera audio usando excitación (impulsos) filtrada por LPC

### 6. De-énfasis
Invierte el pre-énfasis para restaurar balance espectral

## 💡 Consejos

- 🎤 Habla claro y sin ruido de fondo
- 📢 Mantén un volumen moderado
- 🔇 Evita sonidos explosivos (p, t, k) muy cerca del micrófono
- ⏱️ Habla durante toda la grabación (3 segundos)

## 🐛 Solución de Problemas

**Error: "No se encuentra el dispositivo de audio"**
```bash
# Verifica dispositivos disponibles
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**El audio sintetizado suena robótico**
- Es normal, LPC es un algoritmo de baja tasa de bits
- Aumenta el `orden_lpc` a 14-16 para mejor calidad
- Ajusta el período de pitch en la función `sintesis_lpc_frame()`

**Error de memoria**
- Reduce `duracion` a 2 segundos
- Reduce `fs` a 8000 Hz

## 📚 Referencias

- Linear Predictive Coding (LPC) - Teoría de procesamiento de voz
- Algoritmo de Levinson-Durbin - Solución eficiente de sistemas de ecuaciones
- Análisis espectral - Representación tiempo-frecuencia

## 📝 Ejemplo de Salida

```
============================================================
    SINTETIZADOR DE VOZ CON CODIFICACIÓN LPC
============================================================

🎤 Grabando 3 segundos...
¡Habla ahora!
✓ Grabación completada
✓ Guardado: audio_original.wav

🔧 Procesando audio...
✓ Procesamiento completado
✓ Guardado: audio_sintetizado.wav

📊 Generando espectrogramas...
✓ Espectrogramas guardados: espectrogramas.png

🔊 Reproduciendo audio ORIGINAL...
🔊 Reproduciendo audio SINTETIZADO...

============================================================
✓ PROCESO COMPLETADO
============================================================

Archivos generados:
  • audio_original.wav
  • audio_sintetizado.wav
  • espectrogramas.png

¡Listo! 🎉
```
