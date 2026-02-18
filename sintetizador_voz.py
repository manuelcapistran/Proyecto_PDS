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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

class SintetizadorVoz:
    def __init__(self, duracion=3, fs=16000, orden_lpc=12):
        #estos parametros modifican la grabacion de voz

        """
        Inicializa el sintetizador de voz
        
        Parámetros:
        - duracion: segundos de grabación
        - fs: frecuencia de muestreo (Hz)
        - orden_lpc: orden del filtro LPC
        """
        self.duracion = duracion
        self.fs = fs
        self.orden_lpc = orden_lpc
        self.audio_original = None
        self.audio_sintetizado = None
        
    def grabar_audio(self):
        """Graba audio del micrófono"""
        print(f"\n🎤 Grabando {self.duracion} segundos...")
        print("¡Habla ahora!")
        
        audio = sd.rec(int(self.duracion * self.fs), 
                      samplerate=self.fs, 
                      channels=1, 
                      dtype='float64')
        sd.wait()
        
        self.audio_original = audio.flatten()
        print("✓ Grabación completada")
        
    def guardar_wav(self, audio, nombre_archivo):
        """Guarda audio en formato WAV"""
        # Normalizar a rango int16
        audio_normalizado = np.int16(audio / np.max(np.abs(audio)) * 32767)
        wav.write(nombre_archivo, self.fs, audio_normalizado)
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
        # Padding para FFT
        frame_padded = np.concatenate([frame, np.zeros(n)])
        # Autocorrelación vía FFT
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
            # Calcular coeficiente de reflexión
            lambda_i = -np.sum(a[:i] * r[i:0:-1]) / e
            
            # Actualizar coeficientes
            a_new = np.zeros(i + 1)
            a_new[0] = 1.0
            a_new[1:i] = a[1:i] + lambda_i * a[i-1:0:-1]
            a_new[i] = lambda_i
            a[:i+1] = a_new
            
            # Actualizar error
            e = e * (1 - lambda_i**2)
            
        return a, e
    
    def analisis_lpc_frame(self, frame):
        """Realiza análisis LPC en un frame"""
        # 1. Aplicar ventana de Hamming
        ventana = self.ventana_hamming(len(frame))
        frame_ventaneado = frame * ventana
        
        # 2. Calcular autocorrelación
        r = self.autocorrelacion(frame_ventaneado, self.orden_lpc)
        
        # 3. Algoritmo de Levinson-Durbin
        a, e = self.levinson_durbin(r)
        
        # 4. Calcular ganancia
        ganancia = np.sqrt(e)
        
        return a, ganancia
    
    def sintesis_lpc_frame(self, a, ganancia, longitud):
        """Sintetiza un frame usando coeficientes LPC"""
        # Generar excitación (tren de impulsos o ruido)
        # Para voz sonora, usamos impulsos; para no sonora, ruido
        excitacion = np.zeros(longitud)
        
        # Detectar si es sonoro (simplificado: usar energía)
        pitch_period = 80  # Aproximadamente 200 Hz a 16kHz
        
        # Tren de impulsos para frames sonoros
        for i in range(0, longitud, pitch_period):
            if i < longitud:
                excitacion[i] = ganancia
        
        # Síntesis mediante filtrado
        audio_sintetizado = signal.lfilter([1], a, excitacion)
        
        return audio_sintetizado
    
    def procesar_audio(self):
        """Procesa el audio completo con análisis y síntesis LPC"""
        print("\n🔧 Procesando audio...")
        
        # Pre-énfasis
        audio_preenfasis = self.preenfasis(self.audio_original)
        
        # Parámetros de segmentación
        frame_size = int(0.025 * self.fs)  # 25ms
        frame_shift = int(0.010 * self.fs)  # 10ms (overlap de 15ms)
        
        num_frames = (len(audio_preenfasis) - frame_size) // frame_shift + 1
        
        # Inicializar audio sintetizado
        audio_sintetizado = np.zeros(len(audio_preenfasis))
        
        # Procesar cada frame
        for i in range(num_frames):
            inicio = i * frame_shift
            fin = inicio + frame_size
            
            if fin > len(audio_preenfasis):
                break
                
            frame = audio_preenfasis[inicio:fin]
            
            # Análisis LPC
            a, ganancia = self.analisis_lpc_frame(frame)
            
            # Síntesis LPC
            frame_sintetizado = self.sintesis_lpc_frame(a, ganancia, frame_size)
            
            # Overlap-add
            audio_sintetizado[inicio:fin] += frame_sintetizado
        
        # De-énfasis (inverso del pre-énfasis)
        alpha = 0.97
        self.audio_sintetizado = signal.lfilter([1], [1, -alpha], audio_sintetizado)
        
        # Normalizar
        self.audio_sintetizado = self.audio_sintetizado / np.max(np.abs(self.audio_sintetizado))
        
        print("✓ Procesamiento completado")
        
    def calcular_espectrograma(self, audio):
        """Calcula espectrograma"""
        f, t, Sxx = signal.spectrogram(audio, self.fs, 
                                       nperseg=512, 
                                       noverlap=256)
        return f, t, 10 * np.log10(Sxx + 1e-10)  # Convertir a dB
    
    def visualizar_espectrogramas(self, nombre_archivo='espectrogramas.pdf'):
        """Genera visualización de espectrogramas"""
        print("\n📊 Generando espectrogramas...")
        
        # Calcular espectrogramas
        f_orig, t_orig, Sxx_orig = self.calcular_espectrograma(self.audio_original)
        f_sint, t_sint, Sxx_sint = self.calcular_espectrograma(self.audio_sintetizado)
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis de Síntesis de Voz con LPC', fontsize=16, fontweight='bold')
        
        # Señal original
        axes[0, 0].plot(np.linspace(0, self.duracion, len(self.audio_original)), 
                       self.audio_original)
        axes[0, 0].set_title('Señal Original')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Espectrograma original
        im1 = axes[0, 1].pcolormesh(t_orig, f_orig, Sxx_orig, 
                                    shading='gouraud', cmap='viridis')
        axes[0, 1].set_title('Espectrograma - Audio Original')
        axes[0, 1].set_ylabel('Frecuencia (Hz)')
        axes[0, 1].set_xlabel('Tiempo (s)')
        plt.colorbar(im1, ax=axes[0, 1], label='Potencia (dB)')
        
        # Señal sintetizada
        axes[1, 0].plot(np.linspace(0, self.duracion, len(self.audio_sintetizado)), 
                       self.audio_sintetizado, color='orange')
        axes[1, 0].set_title('Señal Sintetizada (LPC)')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Amplitud')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Espectrograma sintetizado
        im2 = axes[1, 1].pcolormesh(t_sint, f_sint, Sxx_sint, 
                                    shading='gouraud', cmap='viridis')
        axes[1, 1].set_title('Espectrograma - Audio Sintetizado')
        axes[1, 1].set_ylabel('Frecuencia (Hz)')
        axes[1, 1].set_xlabel('Tiempo (s)')
        plt.colorbar(im2, ax=axes[1, 1], label='Potencia (dB)')
        
        plt.tight_layout()
        
        # Guardar
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
    
    # Crear sintetizador
    sintetizador = SintetizadorVoz(duracion=3, fs=16000, orden_lpc=12)
    
    # 1. Grabar audio
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
    sintetizador.reproducir_audios()
    
    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  • audio_original.wav")
    print("  • audio_sintetizado.wav")
    print("  • espectrogramas.png")
    print("\n¡Listo! 🎉")


if __name__ == "__main__":
    main()
