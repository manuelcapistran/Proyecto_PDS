#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    print("Instala las dependencias con:\n  pip3 install questionary rich")
    sys.exit(1)

console = Console()

TASAS = [
    questionary.Choice("8000 Hz  — voz telefónica",  value=8000),
    questionary.Choice("16000 Hz — voz estándar",    value=16000),
    questionary.Choice("44100 Hz — calidad CD",      value=44100),
]

MENU = [
    questionary.Choice("🎙   Nueva grabación",    value="grabar"),
    questionary.Choice("📁   Listar grabaciones", value="listar"),
    questionary.Choice("✖    Salir",              value="salir"),
]


def banner():
    console.print(Panel.fit(
        "[bold cyan]🎙  GRABADOR DE AUDIO[/bold cyan]\n"
        "[dim]Procesamiento Digital de Señales[/dim]",
        border_style="cyan",
        padding=(1, 6),
    ))


def limpiar_nombre(nombre):
    nombre = nombre.strip().lower()
    nombre = re.sub(r'\s+', '_', nombre)
    nombre = re.sub(r'[^\w]', '', nombre)
    nombre = re.sub(r'_+', '_', nombre)
    return nombre.strip('_')


def archivos_de(base):
    return {
        'wav':      f"{base}_audio.wav",
        'info':     f"{base}_info.txt",
        'muestras': f"{base}_muestras.txt",
        'grafica':  f"{base}_grafica.png",
    }


def pedir_nombre():
    def validar(texto):
        if not limpiar_nombre(texto):
            return "Nombre inválido. Usa letras, números o guiones bajos."
        return True

    nombre_raw = questionary.text("Nombre base:", validate=validar).ask()
    if nombre_raw is None:
        return None, None

    base = limpiar_nombre(nombre_raw)
    if base != nombre_raw.strip().lower():
        console.print(f"  [dim]→ Ajustado a:[/dim] [bold]{base}[/bold]")

    archivos = archivos_de(base)
    existentes = [f for f in archivos.values() if os.path.exists(f)]

    tabla = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tabla.add_column("", width=2)
    tabla.add_column("")
    for f in archivos.values():
        if f in existentes:
            tabla.add_row("[yellow]⚠[/yellow]", f"[yellow]{f}[/yellow] [dim](ya existe)[/dim]")
        else:
            tabla.add_row("[dim]·[/dim]", f"[dim]{f}[/dim]")

    console.print(Panel(tabla, title="[bold]Archivos a generar[/bold]", border_style="dim"))

    if existentes:
        ok = questionary.confirm("¿Sobreescribir archivos existentes?", default=False).ask()
        if not ok:
            return None, None

    return base, archivos


def grabar_audio(fs):
    console.print(Panel(
        "[bold green]● Grabando[/bold green]  —  Habla ahora\n"
        "[dim]Presiona Enter para detener[/dim]",
        border_style="green",
    ))
    frames = []
    grabando = True
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def callback(indata, frame_count, time_info, status):
        if grabando:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=callback)
    stream.start()
    input()
    grabando = False
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0).flatten()
    duracion = len(audio) / fs
    console.print(f"[green]✓[/green] Completada: [bold]{duracion:.2f} s[/bold]")
    return audio, fecha_hora


def guardar_wav(audio, fs, ruta):
    audio_norm = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wav.write(ruta, fs, audio_norm)
    console.print(f"  [green]✓[/green] {ruta}")


def guardar_info(audio, fs, fecha_hora, ruta):
    duracion = len(audio) / fs
    rms = np.sqrt(np.mean(audio ** 2))
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("=== INFORMACIÓN DE GRABACIÓN ===\n")
        f.write(f"Fecha y hora           : {fecha_hora}\n")
        f.write(f"Frecuencia de muestreo : {fs} Hz\n")
        f.write(f"Total de muestras      : {len(audio)}\n")
        f.write(f"Duración               : {duracion:.2f} segundos\n")
        f.write(f"Amplitud máxima        : {np.max(audio):.6f}\n")
        f.write(f"Amplitud mínima        : {np.min(audio):.6f}\n")
        f.write(f"Volumen promedio (RMS) : {rms:.6f}\n")
    console.print(f"  [green]✓[/green] {ruta}")


def guardar_muestras(audio, ruta):
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("=== MUESTRAS DE AUDIO ===\n")
        for i, m in enumerate(audio):
            f.write(f"{i},{m:.6f}\n")
    console.print(f"  [green]✓[/green] {ruta}")


def graficar_muestras(ruta_txt, ruta_png):
    indices, valores = [], []
    with open(ruta_txt, 'r', encoding='utf-8') as f:
        for linea in f:
            if linea.startswith('=') or not linea.strip():
                continue
            i, v = linea.strip().split(',')
            indices.append(int(i))
            valores.append(float(v))

    plt.figure(figsize=(12, 4))
    plt.plot(indices, valores, linewidth=0.5, color='steelblue')
    plt.title('Muestras de audio')
    plt.xlabel('Muestra')
    plt.ylabel('Amplitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    console.print(f"  [green]✓[/green] {ruta_png}")


def reproducir(audio, fs):
    console.print("\n[cyan]▶[/cyan] Reproduciendo...")
    sd.play(audio, fs)
    sd.wait()
    console.print("[green]✓[/green] Listo")


def flujo_grabacion():
    _, archivos = pedir_nombre()
    if archivos is None:
        return

    fs = questionary.select("Frecuencia de muestreo:", choices=TASAS, default=TASAS[1]).ask()
    if fs is None:
        return

    questionary.press_any_key_to_continue("Presiona Enter para comenzar...").ask()
    audio, fecha_hora = grabar_audio(fs)

    console.print("\n[bold]Guardando archivos...[/bold]")
    guardar_wav(audio, fs, archivos['wav'])
    guardar_info(audio, fs, fecha_hora, archivos['info'])
    guardar_muestras(audio, archivos['muestras'])
    graficar_muestras(archivos['muestras'], archivos['grafica'])

    if questionary.confirm("\n¿Reproducir la grabación?", default=False).ask():
        reproducir(audio, fs)

    questionary.press_any_key_to_continue("\nPresiona Enter para volver al menú...").ask()


def flujo_listar():
    wavs = sorted(f for f in os.listdir('.') if f.endswith('_audio.wav'))
    if not wavs:
        console.print(Panel("[dim]No hay grabaciones guardadas aún.[/dim]", border_style="dim"))
    else:
        tabla = Table(title="Grabaciones", box=box.ROUNDED, border_style="cyan")
        tabla.add_column("Nombre base", style="bold")
        tabla.add_column("Tamaño", justify="right")
        tabla.add_column("Archivos", justify="center")
        for w in wavs:
            base = w.replace('_audio.wav', '')
            size_kb = os.path.getsize(w) / 1024
            presentes = sum(1 for f in archivos_de(base).values() if os.path.exists(f))
            tabla.add_row(base, f"{size_kb:.1f} KB", f"{presentes}/4")
        console.print(tabla)

    questionary.press_any_key_to_continue("\nPresiona Enter para volver al menú...").ask()


def main():
    while True:
        console.clear()
        banner()
        opcion = questionary.select("¿Qué deseas hacer?", choices=MENU).ask()

        if opcion is None or opcion == "salir":
            console.print("\n[dim]Hasta luego.[/dim]\n")
            break
        elif opcion == "grabar":
            flujo_grabacion()
        elif opcion == "listar":
            console.clear()
            banner()
            flujo_listar()


if __name__ == "__main__":
    main()
