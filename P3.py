from collections import Counter
import __main__
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib

from Callback import GrabadorAudio, ReproductorAudio
from Pre import eliminar_silencios_globales, normalizar_audio
from entrenamiento import (
    GaussianMixtureClassifier,
    sr as frecuencia_modelo,
    vectorizar_caracteristicas,
    vectores_caracteristicas as extraer_caracteristicas_entrenamiento,
)

ROOT = Path(__file__).resolve().parent
CARPETA_MODELOS = ROOT / "Modelo"
GRABACIONES_PROCESADAS = ROOT / "grabaciones_procesadas"
sample_rate = frecuencia_modelo
n_fft = 2048
hop_length = 512
duracion_segmento = 10  # Duración de cada segmento en segundos
solapamiento_segmento = 0.5  # Solapamiento entre segmentos (50%)


#======================================
# Opciones de la Interfaz
#======================================

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


#======================================
# Preprocesamiento 
#======================================

def audio_mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio.reshape(-1)

def preprocesar_y_segmentar(audio, duracion, solapamiento, sr):
    if isinstance(audio, (str, Path)):
        audio_np, sr = librosa.load(audio, sr=None, mono=True)
    else:
        audio_np = audio_mono(audio)

    if sr <= 0 or audio_np.size == 0:
        raise ValueError("El audio está vacío o tiene una frecuencia inválida.")
    if not np.isfinite(audio_np).all():
        raise ValueError("El audio contiene valores no válidos.")
    if np.max(np.abs(audio_np)) < 1e-5:
        raise ValueError("El audio está vacío o tiene muy poco volumen.")

    audio_limpio = eliminar_silencios_globales(audio_np, top_db=30)
    if audio_limpio.size == 0 or np.max(np.abs(audio_limpio)) < 1e-5:
        raise ValueError("No se detectó voz después de quitar silencios.")

    GRABACIONES_PROCESADAS.mkdir(parents=True, exist_ok=True)
    ruta_limpia = GRABACIONES_PROCESADAS / (
        f"audio_limpio_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.wav"
    )
    sf.write(ruta_limpia, audio_limpio, sr)

    if sr != sample_rate:
        audio_limpio = librosa.resample(
            audio_limpio,
            orig_sr=sr,
            target_sr=sample_rate,
        )

    duracion_muestras = int(duracion * sample_rate)
    salto_muestras = int(duracion_muestras * (1 - solapamiento))
    segmentos_listos = []

    for inicio in range(0, len(audio_limpio) - duracion_muestras + 1, salto_muestras):
        segmento = audio_limpio[inicio : inicio + duracion_muestras]
        if np.max(np.abs(segmento)) < 0.02:
            continue

        segmentos_listos.append(normalizar_audio(segmento))

    print(f"  -> Generados {len(segmentos_listos)} segmentos de {duracion}s listos para el modelo.")
    return segmentos_listos, ruta_limpia

#======================================
#Extracción de características
#======================================

def extraer_mfcc(audio):
    mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
        )
    mfcc_promedio = np.mean(mfcc, axis=1)
    mfcc_desviacion = np.std(mfcc, axis=1)
    
    return mfcc_promedio, mfcc_desviacion

def extraer_f0(audio):
    # Estimación de la frecuencia fundamental con pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=60,
        fmax=1200,
        sr=sample_rate
    )
    
    f0_valido = []
    for valor in f0:
        if not np.isnan(valor):
            f0_valido.append(valor)
            
    # Si no se detectó voz o f0 vino vacío, devolvemos 0 para evitar error en np.mean
    if len(f0_valido) == 0:
        return 0.0
        
    f0_promedio = np.mean(f0_valido)
    return f0_promedio  # Devolvemos la frecuencia fundamental promedio

def extraer_rms(audio):
    # Calculamos la amplitud RMS por ventana del ejemplo.
    rms = librosa.feature.rms(y = audio)
    rms_promedio = np.mean(rms, axis=1)  # Calculamos el promedio de RMS

    return rms_promedio  # Devolvemos el promedio de RMS

def extraer_zcr(audio):
    zcr = librosa.feature.zero_crossing_rate(y = audio)
    zcr = np.mean(zcr, axis=1)  # Calculamos el promedio de ZCR
    return zcr  # Devolvemos el primer (y único) vector de ZCR

def extraer_centroide(audio):
    centroide = librosa.feature.spectral_centroid(
    # Indicamos la señal de audio que vamos a caracterizar.
    y = audio,
    # Usamos la frecuencia de muestreo original para interpretar las frecuencias.
    sr = sample_rate)  # Devolvemos el primer (y único) vector de centroide
    promedio_centroide = np.mean(centroide, axis=1)  # Calculamos el promedio del centroide
    return promedio_centroide  # Devolvemos el promedio del centroide

def vectores_caracteristicas(audio):
    mfcc_promedio, mfcc_desviacion = extraer_mfcc(audio)
    f0_promedio = extraer_f0(audio)
    rms = extraer_rms(audio)
    zcr = extraer_zcr(audio)
    centroide = extraer_centroide(audio)

    caracteristicas = {
        "mfcc_promedio": mfcc_promedio,
        "mfcc_desviacion": mfcc_desviacion,
        "f0_promedio": f0_promedio,
        "rms": rms,
        "zcr": zcr,
        "centroide": centroide
    }

    return caracteristicas

#======================================
#Modelos de IA
#======================================

def obtener_modelos_disponibles():
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    extensiones_validas = {".pkl", ".joblib"}
    modelos = []

    for elemento in CARPETA_MODELOS.iterdir():
        if elemento.is_file() and elemento.suffix.lower() in extensiones_validas:
            if "encoder" not in elemento.name.lower():
                modelos.append(elemento.name)

    return sorted(modelos)

def cargar_modelo_y_encoder(nombre_modelo):
    ruta_modelo = CARPETA_MODELOS / Path(nombre_modelo).name
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"No se encontró el modelo seleccionado: {ruta_modelo}")

    encoders = [
        elemento
        for elemento in CARPETA_MODELOS.iterdir()
        if elemento.is_file() and "encoder" in elemento.name.casefold()
    ]
    if not encoders:
        raise FileNotFoundError(
            f"No se encontró un archivo de Label Encoder en {CARPETA_MODELOS}."
        )

    if not hasattr(__main__, "GaussianMixtureClassifier"):
        setattr(__main__, "GaussianMixtureClassifier", GaussianMixtureClassifier)
    modelo = joblib.load(ruta_modelo)
    label_encoder = joblib.load(sorted(encoders)[0])
    if not callable(getattr(modelo, "predict", None)):
        raise TypeError(f"El archivo seleccionado no contiene un modelo predictivo: {ruta_modelo}")
    if not callable(getattr(label_encoder, "inverse_transform", None)):
        raise TypeError("El archivo de encoder no permite convertir predicciones a etiquetas.")
    return modelo, label_encoder

def analizar_tipo_voz(audio_crudo, sr, nombre_modelo):
    segmentos, ruta_limpia = preprocesar_y_segmentar(
        audio_crudo, duracion_segmento, solapamiento_segmento, sr
    )
    if not segmentos:
        return {
            "error": (
                f"Se necesitan al menos {duracion_segmento} segundos de voz "
                "para clasificar."
            ),
            "archivo_limpio": str(ruta_limpia),
        }

    modelo, label_encoder = cargar_modelo_y_encoder(nombre_modelo)

    predicciones = []
    f0_valores = []
    mfcc_acum_prom = []
    mfcc_acum_desv = []

    for segmento in segmentos:
        feats = extraer_caracteristicas_entrenamiento(segmento)
        vector_audio = vectorizar_caracteristicas(feats)
        if vector_audio.size != 30:
            raise ValueError(
                f"Se extrajeron {vector_audio.size} características; el modelo requiere 30."
            )

        predicciones.append(modelo.predict(vector_audio.reshape(1, -1))[0])
        f0_valores.append(feats["f0_promedio"])
        mfcc_acum_prom.append(feats["mfcc_promedio"])
        mfcc_acum_desv.append(feats["mfcc_desviacion"])

    prediccion_final = Counter(predicciones).most_common(1)[0][0]
    etiqueta_final = label_encoder.inverse_transform(
        np.asarray([prediccion_final])
    )[0]

    resultado = {
        "tipo": str(etiqueta_final),
        "modelo": nombre_modelo,
        "predicciones_segmentos": [
            str(etiqueta)
            for etiqueta in label_encoder.inverse_transform(np.asarray(predicciones))
        ],
        "archivo_limpio": str(ruta_limpia),
        "f0_mediana": np.median(f0_valores) if f0_valores else 0.0,
        "f0_min": np.min(f0_valores) if f0_valores else 0.0,
        "f0_max": np.max(f0_valores) if f0_valores else 0.0,
        "mfcc_promedio": np.mean(mfcc_acum_prom, axis=0) if mfcc_acum_prom else np.zeros(13),
        "mfcc_desviacion": np.mean(mfcc_acum_desv, axis=0) if mfcc_acum_desv else np.zeros(13)
    }
    return resultado

#======================================
# Visualización
#======================================

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
    fmin = 60
    fmax = 400

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio_trim,
        fmin = fmin,
        fmax = fmax,
        sr = sr,
        frame_length = n_fft,
        hop_length = hop_length)

    tiempo_f0 = librosa.times_like(
        f0,
        sr = sr,
        hop_length = hop_length)

    plt.figure(figsize = (14,4))
    plt.plot(
        tiempo_f0,
        f0,
        marker = "o",
        markersize = 3)

    plt.title('Frecuencia fundamental f0 de "Audio"')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("f0 [Hz]")
    plt.ylim(np.fmin(fmin, fmax), np.fmax(fmin, fmax))
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.show()


def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Laboratorio de voz | MFCC")
    ventana.geometry("980x720")
    ventana.minsize(800, 600)
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
    modelo_var = tk.StringVar()
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
            
        # Actualizar modelos disponibles en la carpeta Modelo
        lista_modelos = obtener_modelos_disponibles()
        combo_modelo["values"] = lista_modelos
        if lista_modelos:
            modelo_recomendado = "RandomForest_regularizado.joblib"
            combo_modelo.set(
                modelo_recomendado if modelo_recomendado in lista_modelos
                else lista_modelos[0]
            )
        else:
            combo_modelo.set("No hay modelos en /Modelo")
            
        estado_var.set(f"{len(entradas)} micrófono(s), {len(salidas)} salida(s) y {len(lista_modelos)} modelo(s) listos.")

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
        else:
            analizar_f0(audio, frecuencia)

    def mostrar_resultado(resultado):
        salida_resultado.config(state="normal")
        salida_resultado.delete("1.0", tk.END)
        if resultado is None:
            salida_resultado.insert(
                tk.END,
                "El análisis terminó, pero no se devolvió un resultado.\n",
            )
        elif resultado.get("error"):
            salida_resultado.insert(tk.END, resultado["error"])
            if resultado.get("archivo_limpio"):
                salida_resultado.insert(
                    tk.END,
                    f"\nAudio limpio guardado en: {resultado['archivo_limpio']}",
                )
        else:
            salida_resultado.insert(
                tk.END,
                f"Modelo usado: {resultado['modelo']}\n"
                f"Tipo de voz predicho: {resultado['tipo']}\n"
                f"Predicciones por segmento: "
                f"{', '.join(resultado['predicciones_segmentos'])}\n"
                f"Audio limpio guardado en: {resultado['archivo_limpio']}\n\n",
            )
            salida_resultado.insert(
                tk.END,
                f"F0 Mediana: {resultado['f0_mediana']:.1f} Hz "
                f"({resultado['f0_min']:.1f} - {resultado['f0_max']:.1f} Hz)\n\n",
            )
            salida_resultado.insert(tk.END, "MFCC promedio:\n")
            salida_resultado.insert(tk.END, f"{np.round(resultado['mfcc_promedio'], 2).tolist()}\n\n")
            salida_resultado.insert(tk.END, "MFCC desviación:\n")
            salida_resultado.insert(tk.END, f"{np.round(resultado['mfcc_desviacion'], 2).tolist()}")
        salida_resultado.config(state="disabled")

    def analizar_referencia():
        if referencia["audio"] is None:
            messagebox.showwarning("Falta audio", "Selecciona primero un audio de referencia.")
            return
        modelo_seleccionado = modelo_var.get()
        if not modelo_seleccionado or modelo_seleccionado.startswith("No hay"):
            messagebox.showwarning("Falta modelo", "Selecciona un modelo válido de la carpeta Modelo.")
            return
            
        try:
            # Pasamos el arreglo numpy que ya tenemos en memoria (referencia["audio"]) y su frecuencia
            resultado = analizar_tipo_voz(referencia["audio"], referencia["sr"], modelo_seleccionado)
            mostrar_resultado(resultado)
            estado_var.set("Análisis de referencia terminado")
        except Exception as error:
            messagebox.showerror("Error de análisis", str(error))

    def iniciar_grabacion():
        if indice_de(entrada_var) is None:
            messagebox.showwarning("Falta micrófono", "Selecciona un dispositivo de entrada.")
            return

        try:
            referencia["grabador"] = GrabadorAudio(
                device=indice_de(entrada_var),
                frecuencia=sample_rate,
                canales=1
            )
            referencia["grabador"].iniciar_grabacion()
            estado_var.set("Grabando...")
            boton_grabar.config(state="disabled")
            boton_detener.config(state="normal")
        except Exception as error:
            messagebox.showerror("Error de grabación", str(error))

    def iniciar_reproduccion():
        if not referencia["ruta"]:
            messagebox.showwarning("Falta audio", "Selecciona una referencia de audio.")
            return
        if indice_de(salida_var) is None:
            messagebox.showwarning("Falta salida", "Selecciona un dispositivo de salida.")
            return

        try:
            referencia["reproductor"] = ReproductorAudio(
                referencia["ruta"],
                device=indice_de(salida_var)
            )
            referencia["reproductor"].reproducir()
            estado_var.set("Reproduciendo...")
        except Exception as error:
            messagebox.showerror("Error de reproducción", str(error))          

    def detener_y_analizar():
        modelo_seleccionado = modelo_var.get()
        if not modelo_seleccionado or modelo_seleccionado.startswith("No hay"):
            messagebox.showwarning("Falta modelo", "Selecciona un modelo válido de la carpeta Modelo.")
            return
            
        try:
            audio = referencia["grabador"].detener_grabacion() if referencia["grabador"] else None
            if referencia["reproductor"]:
                referencia["reproductor"].detener()
            if audio is not None and len(audio) > 0:
                referencia["audio"] = audio
                referencia["sr"] = sample_rate
                resultado = analizar_tipo_voz(audio, sample_rate, modelo_seleccionado)
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

    contenedor = ttk.Frame(ventana, padding=20)
    contenedor.pack(fill="both", expand=True)
    contenedor.columnconfigure(0, weight=1)
    contenedor.columnconfigure(1, weight=1)
    contenedor.rowconfigure(1, weight=1)
    
    ttk.Label(contenedor, text="Laboratorio de voz", style="Titulo.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(
        contenedor,
        text="Selecciona dispositivos, modelo de IA, carga una referencia y evalúa tu voz.",
        style="Ayuda.TLabel",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 15))

    configuracion = ttk.LabelFrame(contenedor, text="Configuración general", padding=15)
    configuracion.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
    configuracion.columnconfigure(0, weight=1)
    
    ttk.Label(configuracion, text="Modelo de IA (Carpeta /Modelo)").grid(row=0, column=0, sticky="w")
    combo_modelo = ttk.Combobox(configuracion, textvariable=modelo_var, state="readonly")
    combo_modelo.grid(row=1, column=0, sticky="ew", pady=(4, 12))

    ttk.Label(configuracion, text="Micrófono").grid(row=2, column=0, sticky="w")
    combo_entrada = ttk.Combobox(configuracion, textvariable=entrada_var, state="readonly")
    combo_entrada.grid(row=3, column=0, sticky="ew", pady=(4, 12))
    
    ttk.Label(configuracion, text="Salida de audio").grid(row=4, column=0, sticky="w")
    combo_salida = ttk.Combobox(configuracion, textvariable=salida_var, state="readonly")
    combo_salida.grid(row=5, column=0, sticky="ew", pady=(4, 12))
    
    ttk.Label(configuracion, text="Referencia").grid(row=6, column=0, sticky="w")
    ttk.Entry(configuracion, textvariable=archivo_var, state="readonly").grid(
        row=7, column=0, sticky="ew", pady=(4, 6)
    )
    ttk.Button(configuracion, text="Elegir audio", command=seleccionar_audio_interfaz).grid(
        row=8, column=0, sticky="ew", pady=(0, 6)
    )
    ttk.Button(configuracion, text="Reproducir", command=iniciar_reproduccion).grid(
        row=9, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Pausar", command=pausar_audio).grid(
        row=10, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Reanudar", command=reanudar_audio).grid(
        row=11, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(configuracion, text="Reiniciar", command=reiniciar_audio).grid(
        row=12, column=0, sticky="ew", pady=(0, 2)
    )
    ttk.Button(
        configuracion,
        text="Actualizar dispositivos y modelos",
        command=actualizar_dispositivos_interfaz,
    ).grid(row=13, column=0, sticky="ew", pady=(8, 0))

    resultados = ttk.LabelFrame(contenedor, text="Análisis", padding=15)
    resultados.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
    resultados.rowconfigure(1, weight=1)
    resultados.columnconfigure(0, weight=1)
    
    ttk.Label(resultados, textvariable=estado_var, style="Ayuda.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, 8)
    )
    salida_resultado = tk.Text(
        resultados, state="disabled", wrap="word", font=("Consolas", 10),
        bg="#17252e", fg="#d8f3ef", relief="flat", padx=10, pady=10,
    )
    salida_resultado.grid(row=1, column=0, sticky="nsew")
    
    ttk.Button(resultados, text="Analizar referencia", command=analizar_referencia).grid(
        row=2, column=0, sticky="ew", pady=(10, 0)
    )
    
    ttk.Label(resultados, text="Gráficas del audio", style="Ayuda.TLabel").grid(
        row=3, column=0, sticky="w", pady=(12, 4)
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
    acciones.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(15, 0))
    acciones.columnconfigure(0, weight=1)
    acciones.columnconfigure(1, weight=1)
    
    global boton_grabar, boton_detener
    boton_grabar = ttk.Button(acciones, text="Grabar y evaluar", style="Accion.TButton", command=iniciar_grabacion)
    boton_grabar.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    boton_detener = ttk.Button(acciones, text="Detener y analizar", command=detener_y_analizar, state="disabled")
    boton_detener.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    actualizar_dispositivos_interfaz()
    ventana.mainloop()


if __name__ == "__main__":
    crear_interfaz()