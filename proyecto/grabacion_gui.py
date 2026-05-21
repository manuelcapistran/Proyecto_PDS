#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont

# ── Paleta ────────────────────────────────────────────────────────────────────
C_BG      = "#0f0f1a"
C_PANEL   = "#1a1a2e"
C_CARD    = "#16213e"
C_ACCENT  = "#4fc3f7"
C_RED     = "#ef5350"
C_GREEN   = "#66bb6a"
C_MUTED   = "#546e7a"
C_TEXT    = "#eceff1"

QSS = f"""
* {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: 'Segoe UI', '.AppleSystemUIFont', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}}
QFrame#header {{
    background-color: {C_PANEL};
}}
QFrame#statusbar {{
    background-color: {C_PANEL};
    border-top: 1px solid #263238;
}}
QFrame#card {{
    background-color: {C_CARD};
    border-radius: 8px;
}}
QLineEdit {{
    background-color: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_MUTED};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}}
QLineEdit:focus {{
    border: 1px solid {C_ACCENT};
}}
QComboBox {{
    background-color: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_MUTED};
    border-radius: 6px;
    padding: 8px 12px;
}}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox QAbstractItemView {{
    background-color: {C_PANEL};
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
    selection-color: #000;
}}
QPushButton {{
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
    color: white;
    border: none;
}}
QPushButton#btn_rec   {{ background-color: {C_RED}; font-size: 15px; }}
QPushButton#btn_rec:hover {{ background-color: #c62828; }}
QPushButton#btn_stop  {{ background-color: #37474f; font-size: 15px; }}
QPushButton#btn_stop:hover {{ background-color: #263238; }}
QPushButton#btn_play  {{ background-color: #1565c0; font-size: 13px; }}
QPushButton#btn_play:hover {{ background-color: #0d47a1; }}
QPushButton#btn_play:disabled {{ background-color: #1a2a3a; color: {C_MUTED}; }}
"""


def limpiar_nombre(nombre: str) -> str:
    nombre = nombre.strip().lower()
    nombre = re.sub(r'\s+', '_', nombre)
    nombre = re.sub(r'[^\w]', '', nombre)
    nombre = re.sub(r'_+', '_', nombre)
    return nombre.strip('_')


def archivos_de(base: str) -> dict:
    return {
        'wav':      f"{base}_audio.wav",
        'info':     f"{base}_info.txt",
        'muestras': f"{base}_muestras.txt",
        'grafica':  f"{base}_grafica.png",
    }


# ── Señales cross-thread ──────────────────────────────────────────────────────
class Bridge(QObject):
    status        = pyqtSignal(str, str)       # mensaje, color
    guardado_ok   = pyqtSignal(dict, float)    # archivos, duracion
    play_done     = pyqtSignal()


# ── Ventana principal ─────────────────────────────────────────────────────────
class GrabadorWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grabador de Audio — PDS")
        self.setFixedSize(700, 800)
        self.setStyleSheet(QSS)

        self._grabando   = False
        self._frames     = []
        self._audio      = None
        self._fs         = 16000
        self._stream     = None
        self._base       = ""
        self._fecha_hora = ""

        self._bridge = Bridge()
        self._bridge.status.connect(self._on_status)
        self._bridge.guardado_ok.connect(self._on_guardado)
        self._bridge.play_done.connect(self._on_play_done)

        self._timer = QTimer()
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick_waveform)

        self._build_ui()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._make_header())
        vbox.addWidget(self._make_body(), stretch=1)
        vbox.addWidget(self._make_statusbar())

    def _make_header(self):
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(90)
        lay = QVBoxLayout(frame)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("🎙  GRABADOR DE AUDIO")
        title.setFont(QFont("Segoe UI" if sys.platform == "win32" else ".AppleSystemUIFont", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C_ACCENT}; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel("Procesamiento Digital de Señales")
        sub.setStyleSheet(f"color: {C_MUTED}; background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(title)
        lay.addWidget(sub)
        return frame

    def _make_body(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(30, 20, 30, 16)
        lay.setSpacing(8)

        # Nombre
        lay.addWidget(self._bold("Nombre base"))
        self._entry = QLineEdit()
        self._entry.setPlaceholderText("ej. voz_prueba")
        self._entry.setFixedHeight(44)
        lay.addWidget(self._entry)

        lay.addSpacing(4)

        # Frecuencia de muestreo
        lay.addWidget(self._bold("Frecuencia de muestreo"))
        self._combo = QComboBox()
        self._combo.addItems([
            "8 000 Hz  — voz telefónica",
            "16 000 Hz — voz estándar",
            "44 100 Hz — calidad CD",
        ])
        self._combo.setCurrentIndex(1)
        self._combo.setFixedHeight(44)
        lay.addWidget(self._combo)

        lay.addSpacing(8)

        # Botón grabar / detener
        self._btn = QPushButton("● INICIAR GRABACIÓN")
        self._btn.setObjectName("btn_rec")
        self._btn.setFixedHeight(54)
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)

        lay.addSpacing(8)

        # Waveform embebido
        self._fig = Figure(figsize=(5.8, 2), dpi=95, facecolor=C_CARD)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(190)
        self._ax_reset()
        lay.addWidget(self._canvas)

        lay.addSpacing(4)

        # Archivos generados
        lay.addWidget(self._bold("Archivos generados"))
        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(14, 10, 14, 10)
        self._lbl_files = QLabel("—")
        self._lbl_files.setStyleSheet(
            f"color: {C_MUTED}; font-family: 'Courier New', monospace; font-size: 12px; background: transparent;"
        )
        card_lay.addWidget(self._lbl_files)
        lay.addWidget(card)

        lay.addSpacing(4)

        # Botón reproducir
        self._btn_play = QPushButton("▶  Reproducir grabación")
        self._btn_play.setObjectName("btn_play")
        self._btn_play.setFixedHeight(42)
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._reproducir)
        lay.addWidget(self._btn_play)

        lay.addStretch()
        return body

    def _make_statusbar(self):
        bar = QFrame()
        bar.setObjectName("statusbar")
        bar.setFixedHeight(34)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        self._lbl_status = QLabel("Listo")
        self._lbl_status.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; background: transparent;")
        lay.addWidget(self._lbl_status)
        return bar

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bold(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI" if sys.platform == "win32" else ".AppleSystemUIFont", 12, QFont.Weight.Bold))
        lbl.setStyleSheet("background: transparent;")
        return lbl

    def _on_status(self, msg, color):
        self._lbl_status.setStyleSheet(f"color: {color}; font-size: 11px; background: transparent;")
        self._lbl_status.setText(msg)

    def _ax_reset(self):
        self._ax.clear()
        self._ax.set_facecolor(C_CARD)
        self._ax.text(0.5, 0.5, "La forma de onda aparecerá aquí",
                      transform=self._ax.transAxes,
                      ha='center', va='center', color=C_MUTED, fontsize=10)
        for sp in self._ax.spines.values():
            sp.set_color("#263238")
        self._ax.tick_params(colors=C_MUTED, labelsize=7)
        self._canvas.draw()

    def _ax_plot(self, data, color=C_ACCENT):
        step = max(1, len(data) // 4000)
        d = data[::step]
        self._ax.clear()
        self._ax.set_facecolor(C_CARD)
        self._ax.plot(d, linewidth=0.6, color=color)
        self._ax.set_xlabel("Muestra", color=C_MUTED, fontsize=7)
        self._ax.set_ylabel("Amplitud", color=C_MUTED, fontsize=7)
        for sp in self._ax.spines.values():
            sp.set_color("#263238")
        self._ax.tick_params(colors=C_MUTED, labelsize=7)
        self._canvas.draw()

    def _get_fs(self):
        return {0: 8000, 1: 16000, 2: 44100}.get(self._combo.currentIndex(), 16000)

    # ── Lógica de grabación ───────────────────────────────────────────────────

    def _toggle(self):
        if not self._grabando:
            self._iniciar()
        else:
            self._detener()

    def _iniciar(self):
        base = limpiar_nombre(self._entry.text())
        if not base:
            self._bridge.status.emit("⚠  Escribe un nombre base válido", C_RED)
            return

        self._base       = base
        self._fs         = self._get_fs()
        self._frames     = []
        self._grabando   = True
        self._fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._btn.setObjectName("btn_stop")
        self._btn.setText("⬛  DETENER GRABACIÓN")
        self._btn.setStyle(self._btn.style())   # refresca QSS
        self._btn_play.setEnabled(False)
        self._entry.setEnabled(False)
        self._combo.setEnabled(False)
        self._lbl_files.setText("—")
        self._lbl_files.setStyleSheet(
            f"color: {C_MUTED}; font-family: 'Courier New', monospace; font-size: 12px; background: transparent;"
        )
        self._ax_reset()
        self._bridge.status.emit("● Grabando...", C_RED)

        def callback(indata, *_):
            if self._grabando:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._fs, channels=1, dtype='float32', callback=callback
        )
        self._stream.start()
        self._timer.start()

    def _tick_waveform(self):
        if self._frames and len(self._frames) > 2:
            data = np.concatenate(self._frames).flatten()
            self._ax_plot(data, color=C_RED)

    def _detener(self):
        self._grabando = False
        self._timer.stop()
        self._stream.stop()
        self._stream.close()

        self._audio  = np.concatenate(self._frames).flatten()
        duracion     = len(self._audio) / self._fs

        self._btn.setObjectName("btn_rec")
        self._btn.setText("● INICIAR GRABACIÓN")
        self._btn.setStyle(self._btn.style())
        self._entry.setEnabled(True)
        self._combo.setEnabled(True)
        self._bridge.status.emit(f"Guardando archivos de '{self._base}'...", C_ACCENT)

        threading.Thread(target=self._guardar_todo, args=(duracion,), daemon=True).start()

    def _guardar_todo(self, duracion):
        archivos = archivos_de(self._base)
        audio, fs = self._audio, self._fs

        audio_norm = np.int16(audio / np.max(np.abs(audio)) * 32767)
        wav.write(archivos['wav'], fs, audio_norm)

        rms = np.sqrt(np.mean(audio ** 2))
        with open(archivos['info'], 'w', encoding='utf-8') as f:
            f.write("=== INFORMACIÓN DE GRABACIÓN ===\n")
            f.write(f"Fecha y hora           : {self._fecha_hora}\n")
            f.write(f"Frecuencia de muestreo : {fs} Hz\n")
            f.write(f"Total de muestras      : {len(audio)}\n")
            f.write(f"Duración               : {duracion:.2f} segundos\n")
            f.write(f"Amplitud máxima        : {np.max(audio):.6f}\n")
            f.write(f"Amplitud mínima        : {np.min(audio):.6f}\n")
            f.write(f"Volumen promedio (RMS) : {rms:.6f}\n")

        with open(archivos['muestras'], 'w', encoding='utf-8') as f:
            f.write("=== MUESTRAS DE AUDIO ===\n")
            for i, m in enumerate(audio):
                f.write(f"{i},{m:.6f}\n")

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(audio, linewidth=0.5, color='steelblue')
        ax2.set_title('Muestras de audio')
        ax2.set_xlabel('Muestra')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(archivos['grafica'], dpi=150)
        plt.close(fig2)

        self._bridge.guardado_ok.emit(archivos, duracion)

    def _on_guardado(self, archivos, duracion):
        self._ax_plot(self._audio, color=C_ACCENT)
        texto = "\n".join(f"✓  {f}" for f in archivos.values())
        self._lbl_files.setText(texto)
        self._lbl_files.setStyleSheet(
            f"color: {C_GREEN}; font-family: 'Courier New', monospace; font-size: 12px; background: transparent;"
        )
        self._btn_play.setEnabled(True)
        self._bridge.status.emit(f"✓  Listo — {duracion:.2f} s grabados en '{self._base}'", C_GREEN)

    def _reproducir(self):
        self._btn_play.setEnabled(False)
        self._bridge.status.emit("▶  Reproduciendo...", C_ACCENT)

        def play():
            sd.play(self._audio, self._fs)
            sd.wait()
            self._bridge.play_done.emit()

        threading.Thread(target=play, daemon=True).start()

    def _on_play_done(self):
        self._btn_play.setEnabled(True)
        self._bridge.status.emit("✓  Reproducción terminada", C_GREEN)


# ── Entrada ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = GrabadorWindow()
    win.show()
    sys.exit(app.exec())
