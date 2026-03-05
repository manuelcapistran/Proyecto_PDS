#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime


def grabar_audio(fs=16000):
    """Graba audio del micrófono hasta que el usuario presione Enter"""
    print("\nGrabación iniciada...")
    print("¡Habla ahora! Presiona Enter para detener la grabación.")

    frames = []
    grabando = True
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def callback(indata, frame_count, time_info, status):
        if grabando:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=fs,
                            channels=1,
                            dtype='float32',
                            callback=callback)
    stream.start()

    input()  # Esperar Enter

    grabando = False
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0).flatten()
    duracion = len(audio) / fs
    print(f"✓ Grabación completada: {duracion:.2f} segundos grabados")

    return audio, fs, fecha_hora


def guardar_wav(audio, fs, nombre_archivo='audio_original.wav'):
    """Guarda audio en formato WAV"""
    audio_normalizado = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wav.write(nombre_archivo, fs, audio_normalizado)
    print(f"✓ Guardado: {nombre_archivo}")


def guardar_info_txt(audio, fs, fecha_hora, nombre_archivo='info_grabacion.txt'):
    """Guarda información general de la grabación"""
    duracion = len(audio) / fs
    rms = np.sqrt(np.mean(audio ** 2))

    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("=== INFORMACIÓN DE GRABACIÓN ===\n")
        f.write(f"Fecha y hora           : {fecha_hora}\n")
        f.write(f"Frecuencia de muestreo : {fs} Hz\n")
        f.write(f"Total de muestras      : {len(audio)}\n")
        f.write(f"Duración               : {duracion:.2f} segundos\n")
        f.write(f"Amplitud máxima        : {np.max(audio):.6f}\n")
        f.write(f"Amplitud mínima        : {np.min(audio):.6f}\n")
        f.write(f"Volumen promedio (RMS) : {rms:.6f}\n")

    print(f"✓ Guardado: {nombre_archivo}")


def guardar_muestras_txt(audio, nombre_archivo='muestras_audio.txt'):
    """Guarda los valores numéricos de cada muestra del audio"""
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("=== MUESTRAS DE AUDIO ===\n")
        for i, muestra in enumerate(audio):
            f.write(f"{i},{muestra:.6f}\n")

    print(f"✓ Guardado: {nombre_archivo}")


if __name__ == "__main__":
    audio, fs, fecha_hora = grabar_audio()
    guardar_wav(audio, fs)
    guardar_info_txt(audio, fs, fecha_hora)
    guardar_muestras_txt(audio)