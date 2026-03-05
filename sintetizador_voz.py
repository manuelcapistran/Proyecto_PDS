#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sintetizador de Voz usando Codificación Predictiva Lineal (LPC)
Incluye: Grabación, Pre-énfasis, Ventana de Hamming, Autocorrelación,
Algoritmo de Levinson-Durbin, Síntesis y Visualización de Espectrogramas
"""

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana, solo guarda archivos
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

class SintetizadorVoz:
    def __init__(self, fs=16000, orden_lpc=12):
        """
        Inicializa el sintetizador de voz

        Parámetros:
        - fs: frecuencia de muestreo (Hz)
        - orden_lpc: orden del filtro LPC
        """
        self.fs = fs
        self.orden_lpc = orden_lpc
        self.audio_original = None
        self.audio_sintetizado = None
        self.duracion = None  # Se calcula tras la grabación

    def grabar_audio(self):
        """Graba audio del micrófono hasta que el usuario presione Enter"""
        print("\n🎤 Grabación iniciada...")
        print("¡Habla ahora! Presiona Enter para detener la grabación.")

        frames = []
        grabando = True

        def callback(indata, frame_count, time_info, status):
            if grabando:
                frames.append(indata.copy())

        stream = sd.InputStream(samplerate=self.fs,
                                channels=1,
                                dtype='float32',
                                callback=callback)
        stream.start()

        input()  # Esperar Enter

        grabando = False
        stream.stop()
        stream.close()

        self.audio_original = np.concatenate(frames, axis=0).flatten().astype('float64')
        self.duracion = len(self.audio_original) / self.fs
        print(f"✓ Grabación completada: {self.duracion:.2f} segundos grabados")

    def guardar_wav(self, audio, nombre_archivo):
        """Guarda audio en formato WAV"""
        audio_normalizado = np.int16(audio / np.max(np.abs(audio)) * 32767)
        wav.write(nombre_archivo, self.fs, audio_normalizado)
        print(f"✓ Guardado: {nombre_archivo}")

    

    def guardar_info_txt(audio, fs, nombre_archivo='info_grabacion.txt'):
        """Guarda información de la grabación en un archivo de texto"""
        duracion = len(audio) / fs
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write("=== INFORMACIÓN DE GRABACIÓN ===\n")
            f.write(f"Frecuencia de muestreo : {fs} Hz\n")
            f.write(f"Total de muestras      : {len(audio)}\n")
            f.write(f"Duración               : {duracion:.2f} segundos\n")
        print(f"✓ Guardado: {nombre_archivo}")
    

    def preenfasis(self, audio, alpha=0.97):
        """Aplica filtro de pre-énfasis"""
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])

    def ventana_hamming(self, longitud):
        """Genera ventana de Hamming"""
        n = np.arange(longitud)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (longitud - 1))

    def autocorrelacion(self, frame, orden):
        """Calcula autocorrelación usando FFT (más eficiente)"""
        n = len(frame)
        frame_padded = np.concatenate([frame, np.zeros(n)])
        fft_frame = np.fft.fft(frame_padded)
        autocorr = np.fft.ifft(fft_frame * np.conj(fft_frame)).real
        return autocorr[:orden+1]

    def levinson_durbin(self, r):
        """
        Algoritmo de Levinson-Durbin para resolver ecuaciones de Yule-Walker

        Parámetros:
        - r: coeficientes de autocorrelación

        Retorna:
        - a: coeficientes del filtro LPC
        - e: error de predicción
        """
        p = len(r) - 1
        a = np.zeros(p + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, p + 1):
            lambda_i = -np.sum(a[:i] * r[i:0:-1]) / e
            a_new = np.zeros(i + 1)
            a_new[0] = 1.0
            a_new[1:i] = a[1:i] + lambda_i * a[i-1:0:-1]
            a_new[i] = lambda_i
            a[:i+1] = a_new
            e = e * (1 - lambda_i**2)

        return a, e

    def analisis_lpc_frame(self, frame):
        """Realiza análisis LPC en un frame"""
        ventana = self.ventana_hamming(len(frame))
        frame_ventaneado = frame * ventana
        r = self.autocorrelacion(frame_ventaneado, self.orden_lpc)
        a, e = self.levinson_durbin(r)
        ganancia = np.sqrt(e)
        return a, ganancia

    def sintesis_lpc_frame(self, a, ganancia, longitud):
        """Sintetiza un frame usando coeficientes LPC"""
        excitacion = np.zeros(longitud)
        pitch_period = 80  # Aproximadamente 200 Hz a 16kHz
        for i in range(0, longitud, pitch_period):
            if i < longitud:
                excitacion[i] = ganancia
        audio_sintetizado = signal.lfilter([1], a, excitacion)
        return audio_sintetizado

    def procesar_audio(self):
        """Procesa el audio completo con análisis y síntesis LPC"""
        print("\n🔧 Procesando audio...")

        audio_preenfasis = self.preenfasis(self.audio_original)
        frame_size = int(0.025 * self.fs)  # 25ms
        frame_shift = int(0.010 * self.fs)  # 10ms
        num_frames = (len(audio_preenfasis) - frame_size) // frame_shift + 1
        audio_sintetizado = np.zeros(len(audio_preenfasis))

        for i in range(num_frames):
            inicio = i * frame_shift
            fin = inicio + frame_size
            if fin > len(audio_preenfasis):
                break
            frame = audio_preenfasis[inicio:fin]
            a, ganancia = self.analisis_lpc_frame(frame)
            frame_sintetizado = self.sintesis_lpc_frame(a, ganancia, frame_size)
            audio_sintetizado[inicio:fin] += frame_sintetizado

        alpha = 0.97
        self.audio_sintetizado = signal.lfilter([1], [1, -alpha], audio_sintetizado)
        self.audio_sintetizado = self.audio_sintetizado / np.max(np.abs(self.audio_sintetizado))
        print("✓ Procesamiento completado")

    def calcular_espectrograma(self, audio):
        """Calcula espectrograma"""
        f, t, Sxx = signal.spectrogram(audio, self.fs, nperseg=512, noverlap=256)
        return f, t, 10 * np.log10(Sxx + 1e-10)

    def visualizar_espectrogramas(self, nombre_archivo='espectrogramas.pdf'):
        """Genera visualización de espectrogramas"""
        print("\n📊 Generando espectrogramas...")

        f_orig, t_orig, Sxx_orig = self.calcular_espectrograma(self.audio_original)
        f_sint, t_sint, Sxx_sint = self.calcular_espectrograma(self.audio_sintetizado)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis de Síntesis de Voz con LPC', fontsize=16, fontweight='bold')

        axes[0, 0].plot(np.linspace(0, self.duracion, len(self.audio_original)), self.audio_original)
        axes[0, 0].set_title('Señal Original')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].grid(True, alpha=0.3)

        im1 = axes[0, 1].pcolormesh(t_orig, f_orig, Sxx_orig, shading='gouraud', cmap='viridis')
        axes[0, 1].set_title('Espectrograma - Audio Original')
        axes[0, 1].set_ylabel('Frecuencia (Hz)')
        axes[0, 1].set_xlabel('Tiempo (s)')
        plt.colorbar(im1, ax=axes[0, 1], label='Potencia (dB)')

        axes[1, 0].plot(np.linspace(0, self.duracion, len(self.audio_sintetizado)), self.audio_sintetizado, color='orange')
        axes[1, 0].set_title('Señal Sintetizada (LPC)')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Amplitud')
        axes[1, 0].grid(True, alpha=0.3)

        im2 = axes[1, 1].pcolormesh(t_sint, f_sint, Sxx_sint, shading='gouraud', cmap='viridis')
        axes[1, 1].set_title('Espectrograma - Audio Sintetizado')
        axes[1, 1].set_ylabel('Frecuencia (Hz)')
        axes[1, 1].set_xlabel('Tiempo (s)')
        plt.colorbar(im2, ax=axes[1, 1], label='Potencia (dB)')

        plt.tight_layout()
        plt.savefig(nombre_archivo.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Espectrogramas guardados: {nombre_archivo.replace('.pdf', '.png')}")
        return fig

    def reproducir_audios(self):
        """Reproduce audio original y sintetizado"""
        print("\n🔊 Reproduciendo audio ORIGINAL...")
        sd.play(self.audio_original, self.fs)
        sd.wait()

        print("🔊 Reproduciendo audio SINTETIZADO...")
        sd.play(self.audio_sintetizado, self.fs)
        sd.wait()


def main():
    print("="*60)
    print("    SINTETIZADOR DE VOZ CON CODIFICACIÓN LPC")
    print("="*60)

    # Crear sintetizador (sin duración fija)
    sintetizador = SintetizadorVoz(fs=16000, orden_lpc=12)

    # 1. Grabar audio (termina con Enter)
    sintetizador.grabar_audio()

    # 2. Guardar audio original
    sintetizador.guardar_wav(sintetizador.audio_original, 'audio_original.wav')

    # 3. Procesar con LPC
    sintetizador.procesar_audio()

    # 4. Guardar audio sintetizado
    sintetizador.guardar_wav(sintetizador.audio_sintetizado, 'audio_sintetizado.wav')

    # 5. Visualizar espectrogramas
    sintetizador.visualizar_espectrogramas('espectrogramas.png')

    # 6. Reproducir ambos audios
    #sintetizador.reproducir_audios()

    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  • audio_original.wav")
    print("  • audio_sintetizado.wav")
    print("  • espectrogramas.png")
    print("\n¡Listo! Programa finalizado")


if __name__ == "__main__":
    main()