#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav


def grabar_audio(fs=16000):
    """Graba audio del micrófono hasta que el usuario presione Enter"""
    print("\nGrabación iniciada...")
    print("¡Habla ahora! Presiona Enter para detener la grabación.")

    frames = []
    grabando = True

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

    return audio, fs


def guardar_wav(audio, fs, nombre_archivo='audio_original.wav'):
    """Guarda audio en formato WAV"""
    audio_normalizado = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wav.write(nombre_archivo, fs, audio_normalizado)
    print(f"✓ Guardado: {nombre_archivo}")


if __name__ == "__main__":
    audio, fs = grabar_audio()
    guardar_wav(audio, fs)
