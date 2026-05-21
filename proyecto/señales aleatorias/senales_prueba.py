#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fs = 16000       # frecuencia de muestreo
duracion = 3     # segundos
t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)


def generar_ruido_gaussiano():
    return np.random.normal(0, 1, len(t))

def generar_ruido_uniforme():
    return np.random.uniform(-1, 1, len(t))

def generar_senoidal(frecuencia=440):
    return np.sin(2 * np.pi * frecuencia * t)

def generar_suma_senoidales(frecuencias=[440, 880, 1320]):
    senal = np.zeros(len(t))
    for f in frecuencias:
        senal += np.sin(2 * np.pi * f * t)
    return senal / np.max(np.abs(senal))  # normalizar

def generar_chirp(f_inicio=100, f_fin=4000):
    return np.sin(2 * np.pi * (f_inicio + (f_fin - f_inicio) * t / duracion) * t)

def generar_pulso_gaussiano(centro=1.5, sigma=0.1):
    return np.exp(-((t - centro) ** 2) / (2 * sigma ** 2))


def guardar_wav(senal, nombre):
    audio = np.int16(senal / np.max(np.abs(senal)) * 32767)
    wav.write(nombre, fs, audio)
    print(f"✓ Guardado: {nombre}")


def graficar_todas(senales, nombres, nombre_png='senales_prueba.png'):
    n = len(senales)
    muestras_visibles = 1000  # mostrar solo las primeras 1000 muestras
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    fig.suptitle('Señales de prueba', fontsize=14)

    for i, (senal, nombre) in enumerate(zip(senales, nombres)):
        axes[i].plot(t[:muestras_visibles], senal[:muestras_visibles], linewidth=0.8, color='steelblue')
        axes[i].set_title(nombre, fontsize=11)
        axes[i].set_xlabel('Tiempo (s)')
        axes[i].set_ylabel('Amplitud')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(nombre_png, dpi=150)
    plt.close()
    print(f"✓ Guardado: {nombre_png}")


if __name__ == "__main__":
    senales = [
        generar_ruido_gaussiano(),
        generar_ruido_uniforme(),
        generar_senoidal(),
        generar_suma_senoidales(),
        generar_chirp(),
        generar_pulso_gaussiano(),
    ]

    nombres = [
        "Ruido blanco gaussiano",
        "Ruido uniforme",
        "Senoidal pura (440 Hz)",
        "Suma de senoidales (440 + 880 + 1320 Hz)",
        "Chirp (100 Hz → 4000 Hz)",
        "Pulso gaussiano",
    ]

    # Guardar cada señal como WAV
    for senal, nombre in zip(senales, nombres):
        guardar_wav(senal, nombre.split('(')[0].strip().lower().replace(' ', '_') + '.wav')

    # Graficar todas juntas
    graficar_todas(senales, nombres)

    print("\n✓ LISTO")