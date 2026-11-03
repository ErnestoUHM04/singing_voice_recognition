from __future__ import annotations

from collections import Counter
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf

from Callback import GrabadorAudio, ReproductorAudio
from preprocesamienot import extraer_caracteristicas

sample_rate = 44100
n_fft = 2048
hop_length = 512
ROOT = Path(__file__).resolve().parent

# Aquí se agarra el modelo entrenado previamente
MODELO_ENTRENADO = ROOT / "randomforest_200_arboles_profundidad_10_resultados" / "modelo.joblib"
GRABACIONES_PROCESADAS = ROOT / "grabaciones_procesadas"
SEGMENTO_SEGUNDOS = 30
MAX_SEGMENTOS_GRABACION = 3
UMBRAL_SILENCIO_DB = 30
_modelo_cache = None

RANGOS_VOCALES = {
    "Bajo": (82, 330),
    "Baritono": (98, 392),
    "Tenor": (131, 523),
    "Contralto": (175, 698),
    "Mezzosoprano": (220, 880),
    "Soprano": (262, 1047),
}


def elegir_salida():
    dispositivos = sd.query_devices()
    salidas = []
    for i, dispositivo in enumerate(dispositivos):
        if dispositivo["max_output_channels"] > 0:
            salidas.append((i, dispositivo["name"]))
    return salidas


def elegir_entrada():
    dispositivos = sd.query_devices()
    entradas = []
    for i, dispositivo in enumerate(dispositivos):
        if dispositivo["max_input_channels"] > 0:
            entradas.append((i, dispositivo["name"]))
    return entradas

def actualizar_dispositivos():
    entradas = elegir_entrada()
    salidas = elegir_salida()
    return entradas, salidas

def cargar_audio():
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Seleccione el audio de referencia",
        filetypes=[
            ("Archivos de audio", "*.wav"),
            ("Archivos WAV", "*.wav"),
        ],
    )
    root.destroy()

    if not ruta:
        print("No se selecciono ningun archivo.")
        return None

    return ruta


def audio_mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio.reshape(-1)


def cargar_modelo_entrenado():
    global _modelo_cache
    if _modelo_cache is None:
        if not MODELO_ENTRENADO.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo seleccionado: {MODELO_ENTRENADO}"
            )
        paquete = joblib.load(MODELO_ENTRENADO)
        if "modelo" not in paquete or "caracteristicas" not in paquete:
            raise ValueError("El archivo no contiene el formato de modelo esperado.")
        _modelo_cache = paquete
    return _modelo_cache


def procesar_grabacion_con_modelo(audio, sr, prefijo_salida="grabacion"):
    audio = np.nan_to_num(audio_mono(audio))
    if sr <= 0 or audio.size == 0 or np.max(np.abs(audio)) < 1e-5:
        return {"error": "La grabación está vacía o tiene muy poco volumen."}, audio
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    audio_limpio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.8,
        n_jobs=1,
    )
    audio_limpio = np.nan_to_num(audio_limpio).astype(np.float32)
    intervalos = librosa.effects.split(audio_limpio, top_db=UMBRAL_SILENCIO_DB)
    if len(intervalos) == 0:
        return {"error": "No se detectó voz después de limpiar el audio."}, audio_limpio
    voz_sin_silencios = np.concatenate(
        [audio_limpio[inicio:fin] for inicio, fin in intervalos]
    )
    if np.max(np.abs(voz_sin_silencios)) < 1e-5:
        return {"error": "No se detectó voz después de quitar silencios."}, voz_sin_silencios

    GRABACIONES_PROCESADAS.mkdir(parents=True, exist_ok=True)
    ruta_limpia = GRABACIONES_PROCESADAS / (
        f"{prefijo_salida}_limpio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    )
    sf.write(ruta_limpia, voz_sin_silencios, sr)

    paquete = cargar_modelo_entrenado()
    columnas_modelo = paquete["caracteristicas"]
    tamanio_segmento = SEGMENTO_SEGUNDOS * sr
    predicciones = []
    caracteristicas_segmentos = []

    for indice in range(MAX_SEGMENTOS_GRABACION):
        inicio = indice * tamanio_segmento
        if inicio >= len(voz_sin_silencios):
            break
        segmento = voz_sin_silencios[inicio : inicio + tamanio_segmento]

        caracteristicas = extraer_caracteristicas(segmento, sr)
        faltantes = set(columnas_modelo) - caracteristicas.keys()
        if faltantes:
            raise ValueError(
                f"Faltan características requeridas por el modelo: {sorted(faltantes)}"
            )
        entrada_modelo = pd.DataFrame(
            [{nombre: caracteristicas[nombre] for nombre in columnas_modelo}],
            columns=columnas_modelo,
        )
        predicciones.append(paquete["modelo"].predict(entrada_modelo)[0])
        caracteristicas_segmentos.append(caracteristicas)

    if not predicciones:
        return {
            "error": "No se generaron segmentos de voz para clasificar."
        }, voz_sin_silencios

    tipo_predicho = Counter(predicciones).most_common(1)[0][0]
    promedio = lambda nombre: float(
        np.mean([segmento[nombre] for segmento in caracteristicas_segmentos])
    )
    mfcc_promedio = np.array(
        [promedio(f"MFCC_{indice:02d}_media") for indice in range(1, 14)]
    )
    mfcc_desviacion = np.array(
        [promedio(f"MFCC_{indice:02d}_desv") for indice in range(1, 14)]
    )
    resultado = {
        "tipo": tipo_predicho,
        "f0_mediana": promedio("F0_media"),
        "f0_min": min(segmento["Altura_min_Hz"] for segmento in caracteristicas_segmentos),
        "f0_max": max(segmento["Altura_max_Hz"] for segmento in caracteristicas_segmentos),
        "mfcc_promedio": mfcc_promedio,
        "mfcc_desviacion": mfcc_desviacion,
        "predicciones_segmentos": predicciones,
        "archivo_limpio": str(ruta_limpia),
    }
    return resultado, voz_sin_silencios


def mostrar_forma_onda(audio, sr):
    audio = audio_mono(audio)
    tiempo = np.arange(len(audio)) / sr
    plt.figure(figsize=(14, 4))
    plt.plot(tiempo, audio)
    plt.title('Forma de onda de la grabacion "Audio"')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def mostrar_espectrograma(audio, sr):
    audio = audio_mono(audio)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitud = np.abs(stft)
    espectrograma_db = librosa.amplitude_to_db(magnitud, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        espectrograma_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title('Espectrograma de la grabacion "Audio"')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.tight_layout()
    plt.show()


def mostrar_espectrograma_mel(audio, sr):
    audio = audio_mono(audio)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title('Espectrograma Mel de la grabacion "Audio"')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.tight_layout()
    plt.show()

def analizar_f0(audio, sr):
    audio = audio_mono(audio)
    audio_trim, _ = librosa.effects.trim(audio)
 # Limitamos la búsqueda de frecuencia fundamental a 60–400 Hz.
# Este es el intervalo elegido para el análisis, no todo el rango posible de la voz humana.
    fmin = 60
    fmax = 400 # <------ ESTA VARIABLE SE DEBERÍA DE ACTUALIZAR DE ACUERDO AL RANGO DEL AUDIO

# Estimamos f0 con pYIN usando las mismas ventanas del análisis anterior.
# voiced_flag indica las ventanas clasificadas como sonoras y voiced_prob su probabilidad.
# Por defecto, f0 contiene NaN en las ventanas clasificadas como no sonoras.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio_trim,
        fmin = fmin,
        fmax = fmax,
        sr = sr,
        frame_length = n_fft,
        hop_length = hop_length)

# Asignamos un instante a cada estimación según la frecuencia de muestreo y el salto.
    tiempo_f0 = librosa.times_like(
        f0,
        sr = sr,
        hop_length = hop_length)

    plt.figure(figsize = (14,4))

# Mostramos las estimaciones con puntos; los valores NaN dejan huecos en la curva.
    plt.plot(
        tiempo_f0,
        f0,
        marker = "o",
        markersize = 3)

    plt.title('Frecuencia fundamental f0 de "Audio"')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("f0 [Hz]")

# Ajustamos el eje vertical al intervalo de búsqueda, sin modificar los valores calculados.
    plt.ylim(np.fmin(fmin, fmax), np.fmax(fmin, fmax))
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.show()


def analizar_tipo_voz(audio, sr=44100):
    audio = np.asarray(audio, dtype=np.float32).flatten()

    if len(audio) == 0:
        return {"error": "No hay audio para analizar."}

    if np.max(np.abs(audio)) < 1e-5:
        return {"error": "La señal está vacía o tiene muy poco volumen."}

    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=512,
        n_mels=128,
    )
    mfcc_promedio = np.mean(mfcc, axis=1)
    mfcc_desviacion = np.std(mfcc, axis=1)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=65,
        fmax=1100,
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    f0_validas = f0[np.isfinite(f0)]

    if len(f0_validas) < 5:
        return {
            "error": "No se detectaron suficientes segmentos sonoros para estimar el rango vocal.",
            "mfcc_promedio": mfcc_promedio,
            "mfcc_desviacion": mfcc_desviacion,
        }

    f0_min = float(np.percentile(f0_validas, 5))
    f0_max = float(np.percentile(f0_validas, 95))
    f0_mediana = float(np.median(f0_validas))

    puntuaciones = {}
    for tipo, (limite_min, limite_max) in RANGOS_VOCALES.items():
        dentro = (f0_validas >= limite_min) & (f0_validas <= limite_max)
        porcentaje = 100 * np.mean(dentro)
        cobertura = (
            max(0, min(f0_max, limite_max) - max(f0_min, limite_min))
            / max(f0_max - f0_min, 1e-9)
        )
        puntuaciones[tipo] = {
            "porcentaje_f0": float(porcentaje),
            "cobertura_rango": float(cobertura * 100),
        }

    tipo_estimado = max(
        puntuaciones,
        key=lambda tipo: (
            puntuaciones[tipo]["porcentaje_f0"],
            puntuaciones[tipo]["cobertura_rango"],
        ),
    )

    return {
        "tipo": tipo_estimado,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_mediana": f0_mediana,
        "mfcc_promedio": mfcc_promedio,
        "mfcc_desviacion": mfcc_desviacion,
        "puntuaciones": puntuaciones,
    }


def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Laboratorio de voz | MFCC")
    ventana.geometry("980x650")
    ventana.minsize(800, 540)
    ventana.configure(bg="#eef2f5")

    estilo = ttk.Style(ventana)
    estilo.theme_use("clam")
    estilo.configure("TFrame", background="#eef2f5")
    estilo.configure("TLabel", background="#eef2f5", foreground="#243746")
    estilo.configure("Titulo.TLabel", font=("Segoe UI", 22, "bold"), foreground="#17324d")
    estilo.configure("Ayuda.TLabel", foreground="#647783")
    estilo.configure("Accion.TButton", background="#167d8d", foreground="white", padding=9)
    estilo.map("Accion.TButton", background=[("active", "#105b68")])

    referencia = {"ruta": None, "reproductor": None, "grabador": None, "audio": None, "sr": sample_rate}
    entrada_var = tk.StringVar()
    salida_var = tk.StringVar()
    archivo_var = tk.StringVar(value="Ningún archivo seleccionado")
    estado_var = tk.StringVar(value="Listo para comenzar")

    def indice_de(variable):
        valor = variable.get().split(":", 1)[0].strip()
        return int(valor) if valor else None

    def actualizar_dispositivos_interfaz():
        entradas, salidas = actualizar_dispositivos()
        combo_entrada["values"] = [f"{i}: {nombre}" for i, nombre in entradas]
        combo_salida["values"] = [f"{i}: {nombre}" for i, nombre in salidas]
        if entradas:
            combo_entrada.current(0)
        if salidas:
            combo_salida.current(0)
        estado_var.set(f"{len(entradas)} micrófono(s) y {len(salidas)} salida(s) disponibles")

    def seleccionar_audio_interfaz():
        ruta = cargar_audio()
        if ruta:
            try:
                audio, frecuencia = sf.read(ruta, dtype="float32")
                referencia["ruta"] = ruta
                referencia["audio"] = audio
                referencia["sr"] = frecuencia
                archivo_var.set(ruta)
                estado_var.set("Audio de referencia cargado")
            except Exception as error:
                messagebox.showerror("Audio inválido", str(error))

    def mostrar_grafica_existente(tipo):
        if referencia["audio"] is None:
            messagebox.showwarning("Falta audio", "Carga o graba un audio antes de mostrar una gráfica.")
            return

        audio = referencia["audio"]
        frecuencia = referencia["sr"]
        if tipo == "onda":
            mostrar_forma_onda(audio, frecuencia)
        elif tipo == "espectrograma":
            mostrar_espectrograma(audio, frecuencia)
        elif tipo == "mel":
            mostrar_espectrograma_mel(audio, frecuencia)
        else :
            analizar_f0(audio, frecuencia)


    def mostrar_resultado(resultado):
        salida_resultado.config(state="normal")
        salida_resultado.delete("1.0", tk.END)
        if resultado is None:
            salida_resultado.insert(
                tk.END,
                "El análisis terminó, pero analizar_tipo_voz no devolvió un resultado.\n"
                "Revisa la función de análisis para retornar sus puntuaciones.",
            )
        elif resultado.get("error"):
            salida_resultado.insert(tk.END, resultado["error"])
        else:
            salida_resultado.insert(tk.END, f"Tipo de voz estimado: {resultado['tipo']}\n")
            salida_resultado.insert(
                tk.END,
                f"F0: {resultado['f0_mediana']:.1f} Hz "
                f"({resultado['f0_min']:.1f} - {resultado['f0_max']:.1f} Hz)\n\n",
            )
            salida_resultado.insert(tk.END, "MFCC promedio:\n")
            salida_resultado.insert(tk.END, f"{np.round(resultado['mfcc_promedio'], 2).tolist()}\n\n")
            salida_resultado.insert(tk.END, "MFCC desviación:\n")
            salida_resultado.insert(tk.END, f"{np.round(resultado['mfcc_desviacion'], 2).tolist()}")
        salida_resultado.config(state="disabled")

    def analizar_referencia():
        if not referencia["ruta"]:
            messagebox.showwarning("Falta audio", "Selecciona primero un audio de referencia.")
            return
        try:
            audio, frecuencia = sf.read(referencia["ruta"], dtype="float32")
            resultado, audio_limpio = procesar_grabacion_con_modelo(
                audio,
                frecuencia,
                prefijo_salida="referencia",
            )
            referencia["audio"] = audio_limpio
            referencia["sr"] = sample_rate
            mostrar_resultado(resultado)
            estado_var.set("Análisis de referencia terminado")
        except Exception as error:
            messagebox.showerror("Error de análisis", str(error))

    def iniciar_grabacion():
        if indice_de(entrada_var) is None:
            messagebox.showwarning(
                "Falta micrófono",
                "Selecciona un dispositivo de entrada."
            )
            return

        try:
            referencia["grabador"] = GrabadorAudio(
                device=indice_de(entrada_var),
                frecuencia=sample_rate,
                canales=1
            )

            referencia["grabador"].iniciar_grabacion()

            estado_var.set("Grabando")

            boton_grabar.config(state="disabled")
            boton_detener.config(state="normal")

        except Exception as error:
            messagebox.showerror(
                "Error de grabación",
                str(error)
            )

    def iniciar_reproduccion():
        if not referencia["ruta"]:
            messagebox.showwarning(
                "Falta audio",
                "Selecciona una referencia de audio."
            )
            return

        if indice_de(salida_var) is None:
            messagebox.showwarning(
                "Falta salida",
                "Selecciona un dispositivo de salida."
            )
            return

        try:
            referencia["reproductor"] = ReproductorAudio(
                referencia["ruta"],
                device=indice_de(salida_var)
            )

            referencia["reproductor"].reproducir()

            estado_var.set("Reproduciendo")

        except Exception as error:
            messagebox.showerror(
                "Error de reproducción",
                str(error)
            )          

    def detener_y_analizar():
        try:
            audio = referencia["grabador"].detener_grabacion() if referencia["grabador"] else None
            if referencia["reproductor"]:
                referencia["reproductor"].detener()
            if audio is not None and len(audio) > 0:
                referencia["sr"] = sample_rate
                resultado, audio_limpio = procesar_grabacion_con_modelo(audio, sample_rate)
                referencia["audio"] = audio_limpio
                mostrar_resultado(resultado)
                estado_var.set("Grabación analizada")
            else:
                estado_var.set("Grabación detenida sin audio")
        except Exception as error:
            messagebox.showerror("Error de análisis", str(error))
        finally:
            boton_grabar.config(state="normal")
            boton_detener.config(state="disabled")

    def pausar_audio():
        if referencia["reproductor"]:
            referencia["reproductor"].pausar()
            estado_var.set("Reproducción pausada")

    def reanudar_audio():
        if referencia["reproductor"]:
            referencia["reproductor"].reanudar()
            estado_var.set("Reproducción reanudada")

    def reiniciar_audio():
        if referencia["reproductor"]:
            referencia["reproductor"].detener()
            referencia["reproductor"].reproducir()
            estado_var.set("Reproducción reiniciada")

    contenedor = ttk.Frame(ventana, padding=26)
    contenedor.pack(fill="both", expand=True)
    contenedor.columnconfigure(0, weight=1)
    contenedor.columnconfigure(1, weight=1)
    contenedor.rowconfigure(1, weight=1)
    ttk.Label(contenedor, text="Laboratorio de voz", style="Titulo.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(
        contenedor,
        text="Selecciona tus dispositivos, carga una referencia y evalúa tu voz con MFCC.",
        style="Ayuda.TLabel",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 22))

    configuracion = ttk.LabelFrame(contenedor, text="Configuración de audio", padding=18)
    configuracion.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
    configuracion.columnconfigure(0, weight=1)
    ttk.Label(configuracion, text="Micrófono").grid(row=0, column=0, sticky="w")
    combo_entrada = ttk.Combobox(configuracion, textvariable=entrada_var, state="readonly")
    combo_entrada.grid(row=1, column=0, sticky="ew", pady=(4, 16))
    ttk.Label(configuracion, text="Salida de audio").grid(row=2, column=0, sticky="w")
    combo_salida = ttk.Combobox(configuracion, textvariable=salida_var, state="readonly")
    combo_salida.grid(row=3, column=0, sticky="ew", pady=(4, 16))
    ttk.Label(configuracion, text="Referencia").grid(row=4, column=0, sticky="w")
    ttk.Entry(configuracion, textvariable=archivo_var, state="readonly").grid(
        row=5, column=0, sticky="ew", pady=(4, 8)
    )
    ttk.Button(configuracion, text="Elegir audio", command=seleccionar_audio_interfaz).grid(
        row=6, column=0, sticky="ew", pady=(0, 8)
    )
    ttk.Button(configuracion, text="Reproducir", command=iniciar_reproduccion).grid(
        row=7, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Pausar", command=pausar_audio).grid(
        row=8, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Reanudar", command=reanudar_audio).grid(
        row=9, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Reiniciar", command=reiniciar_audio).grid(
        row=10, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(
        configuracion,
        text="Actualizar dispositivos",
        command=actualizar_dispositivos_interfaz,
    ).grid(row=11, column=0, columnspan=4, sticky="ew")

    resultados = ttk.LabelFrame(contenedor, text="Análisis", padding=18)
    resultados.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
    resultados.rowconfigure(1, weight=1)
    resultados.columnconfigure(0, weight=1)
    ttk.Label(resultados, textvariable=estado_var, style="Ayuda.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, 10)
    )
    salida_resultado = tk.Text(
        resultados, state="disabled", wrap="word", font=("Consolas", 10),
        bg="#17252e", fg="#d8f3ef", relief="flat", padx=14, pady=14,
    )
    salida_resultado.grid(row=1, column=0, sticky="nsew")
    ttk.Button(resultados, text="Analizar referencia", command=analizar_referencia).grid(
        row=2, column=0, sticky="ew", pady=(12, 0)
    )
    ttk.Label(resultados, text="Gráficas del audio", style="Ayuda.TLabel").grid(
        row=3, column=0, sticky="w", pady=(18, 6)
    )
    graficas = ttk.Frame(resultados)
    graficas.grid(row=4, column=0, sticky="ew")
    ttk.Button(graficas, text="Onda", command=lambda: mostrar_grafica_existente("onda")).pack(
        side="left", padx=(0, 4)
    )
    ttk.Button(
        graficas,
        text="Espectro",
        command=lambda: mostrar_grafica_existente("espectrograma"),
    ).pack(side="left", padx=4)
    ttk.Button(graficas, text="Mel", command=lambda: mostrar_grafica_existente("mel")).pack(
        side="left", padx=4
    )
    ttk.Button(graficas, text="F0", command=lambda: mostrar_grafica_existente("F0")).pack(
            side="left", padx=4
        )

    acciones = ttk.Frame(contenedor)
    acciones.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(18, 0))
    acciones.columnconfigure(0, weight=1)
    acciones.columnconfigure(1, weight=1)
    boton_grabar = ttk.Button(acciones, text="Grabar y evaluar", style="Accion.TButton", command=iniciar_grabacion)
    boton_grabar.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    boton_detener = ttk.Button(acciones, text="Detener y analizar", command=detener_y_analizar, state="disabled")
    boton_detener.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    actualizar_dispositivos_interfaz()
    ventana.mainloop()


if __name__ == "__main__":
    crear_interfaz()
