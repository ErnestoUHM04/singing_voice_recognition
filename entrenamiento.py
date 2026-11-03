import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import librosa
import numpy as np
from pathlib import Path
import os
import sys
import tempfile
import traceback
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import joblib

sr= 22050  # Frecuencia de muestreo estándar para audio
n_fft = 2048  # Tamaño de la ventana de análisis
hop_length = 512  # Desplazamiento entre ventanas
MAX_HILOS_EXTRACCION = min(6, max(1, (os.cpu_count() or 2) - 2))
COLUMNAS_CARACTERISTICAS = [
    *(f"mfcc_promedio_{indice:02d}" for indice in range(1, 14)),
    *(f"mfcc_desviacion_{indice:02d}" for indice in range(1, 14)),
    "f0_promedio",
    "rms",
    "zcr",
    "centroide",
]
COLUMNAS_CACHE = [
    "archivo",
    "etiqueta",
    "grupo",
    "tamano_bytes",
    "mtime_ns",
    "incluida",
    *COLUMNAS_CARACTERISTICAS,
]


class GaussianMixtureClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-3,
        random_state=42,
        max_iter=100,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X e y deben contener el mismo número de ejemplos válidos.")

        self.classes_ = np.unique(y)
        self.models_ = {}
        for clase in self.classes_:
            ejemplos_clase = X[y == clase]
            modelo = GaussianMixture(
                n_components=min(self.n_components, len(ejemplos_clase)),
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=1,
            )
            modelo.fit(ejemplos_clase)
            self.models_[clase] = modelo
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        puntuaciones = np.column_stack(
            [self.models_[clase].score_samples(X) for clase in self.classes_]
        )
        return self.classes_[np.argmax(puntuaciones, axis=1)]


class SalidaDuplicada:
    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for destino in self.destinos:
            destino.write(texto)
        return len(texto)

    def flush(self):
        for destino in self.destinos:
            destino.flush()


def extraer_mfcc(audio):
    mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
        )
    mfcc_promedio = np.mean(mfcc, axis=1)
    mfcc_desviacion = np.std(mfcc, axis=1)
    
    return mfcc_promedio, mfcc_desviacion

def extraer_f0(audio):
    f0, voiced_flag, voiced_probs = librosa.pyin(
    # Proporcionamos la señal preprocesada del ejemplo.
        audio,
    # Establecemos en 50 Hz el límite inferior para estimar F0.
        fmin = 60,
    # Establecemos en 500 Hz el límite superior para estimar F0.
        fmax = 1200,
    # Usamos la frecuencia de muestreo original para interpretar las frecuencias.
        sr = sr,)
    f0_validas = f0[np.isfinite(f0)]
    if f0_validas.size == 0:
        return np.nan

    return np.mean(f0_validas)

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
    sr = sr)  # Devolvemos el primer (y único) vector de centroide
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


def vectorizar_caracteristicas(caracteristicas):
    vector = np.hstack(
        [
            caracteristicas["mfcc_promedio"],
            caracteristicas["mfcc_desviacion"],
            [caracteristicas["f0_promedio"]],
            caracteristicas["rms"],
            caracteristicas["zcr"],
            caracteristicas["centroide"],
        ]
    )
    if vector.size != len(COLUMNAS_CARACTERISTICAS):
        raise ValueError(
            f"La extracción produjo {vector.size} características; "
            f"se esperaban {len(COLUMNAS_CARACTERISTICAS)}."
        )
    return vector


def metadatos_archivo(archivo):
    stat = archivo.stat()
    return {
        "archivo": archivo.name,
        "etiqueta": archivo.stem.split("_")[0],
        "grupo": archivo.stem.split("_seg_")[0],
        "tamano_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def extraer_fila_caracteristicas(archivo):
    audio, _ = librosa.load(archivo, sr=sr)
    caracteristicas = vectores_caracteristicas(audio)
    vector = vectorizar_caracteristicas(caracteristicas)
    incluida = bool(np.isfinite(vector).any())
    vector[~np.isfinite(vector)] = np.nan
    if not incluida:
        vector[:] = np.nan
    return metadatos_archivo(archivo), vector, incluida


def cargar_cache_caracteristicas(ruta_csv, archivos):
    if not ruta_csv.is_file() or ruta_csv.stat().st_size == 0:
        return None

    try:
        with ruta_csv.open("r", newline="", encoding="utf-8") as archivo_csv:
            lector = csv.DictReader(archivo_csv)
            if lector.fieldnames != COLUMNAS_CACHE:
                raise ValueError("El esquema del CSV no coincide.")
            filas = list(lector)

        if len(filas) != len(archivos):
            raise ValueError("El número de filas del CSV no coincide con los audios.")

        matriz = []
        etiquetas = []
        grupos = []
        for fila, archivo in zip(filas, archivos):
            for campo, valor_actual in metadatos_archivo(archivo).items():
                if fila[campo] != str(valor_actual):
                    raise ValueError(
                        f"El CSV está desactualizado para {archivo.name}."
                    )

            vector = np.asarray(
                [float(fila[columna]) for columna in COLUMNAS_CARACTERISTICAS],
                dtype=float,
            )
            if fila["incluida"] == "1":
                if not np.isfinite(vector).any():
                    raise ValueError(
                        f"Las características de {archivo.name} no son válidas."
                    )
                matriz.append(vector)
                etiquetas.append(fila["etiqueta"])
                grupos.append(fila["grupo"])
            elif fila["incluida"] != "0":
                raise ValueError("El CSV contiene un indicador de inclusión inválido.")

        print(f"Características reutilizadas desde: {ruta_csv}", flush=True)
        return np.asarray(matriz), np.asarray(etiquetas), np.asarray(grupos)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(
            f"No se pudo reutilizar el CSV de características ({error}); "
            "se recalculará.",
            flush=True,
        )
        return None


def guardar_cache_caracteristicas(ruta_csv, filas_cache):
    ruta_temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=ruta_csv.parent,
            suffix=".tmp",
            delete=False,
        ) as archivo_temporal:
            ruta_temporal = Path(archivo_temporal.name)
            escritor = csv.DictWriter(archivo_temporal, fieldnames=COLUMNAS_CACHE)
            escritor.writeheader()
            escritor.writerows(filas_cache)
        os.replace(ruta_temporal, ruta_csv)
        print(f"Características guardadas en: {ruta_csv}", flush=True)
    finally:
        if ruta_temporal is not None and ruta_temporal.exists():
            ruta_temporal.unlink()


def crear_matriz_caracteristicas(carpeta_dataset):
    carpeta = Path(carpeta_dataset)
    if not carpeta.is_absolute():
        carpeta = Path(__file__).resolve().parent / carpeta
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta del dataset: {carpeta}")

    archivos = sorted(carpeta.glob("*.wav"))
    if len(archivos) == 0:
        print(f"No se encontraron archivos .wav en la carpeta '{carpeta_dataset}'.")
        return None, None, None

    ruta_csv = carpeta / "caracteristicas.csv"
    cache = cargar_cache_caracteristicas(ruta_csv, archivos)
    if cache is not None:
        return cache

    print(
        f"Extrayendo características para {len(archivos)} audios...",
        flush=True,
    )
    resultados = [None] * len(archivos)
    with ThreadPoolExecutor(max_workers=MAX_HILOS_EXTRACCION) as executor:
        trabajos = {
            executor.submit(extraer_fila_caracteristicas, archivo): indice
            for indice, archivo in enumerate(archivos)
        }
        for completados, trabajo in enumerate(as_completed(trabajos), start=1):
            indice = trabajos[trabajo]
            resultados[indice] = trabajo.result()
            if (
                completados == 1
                or completados % 100 == 0
                or completados == len(archivos)
            ):
                print(
                    f"Características extraídas: {completados}/{len(archivos)}",
                    flush=True,
                )

    X = []
    y = []
    grupos = []
    filas_cache = []
    filas_sin_datos = 0
    valores_no_finitos = 0
    for archivo, resultado in zip(archivos, resultados):
        metadatos, vector_audio, incluida = resultado
        etiqueta = metadatos["etiqueta"]
        mascara_finita = np.isfinite(vector_audio)
        if not incluida:
            filas_sin_datos += 1
            print(f"Audio omitido: no produjo características válidas ({archivo.name})")
            filas_cache.append(
                {
                    **metadatos,
                    "incluida": 0,
                    **dict(zip(COLUMNAS_CARACTERISTICAS, vector_audio)),
                }
            )
            continue

        valores_no_finitos += int((~mascara_finita).sum())
        vector_audio[~mascara_finita] = np.nan
        
        X.append(vector_audio)
        y.append(etiqueta)
        grupo = metadatos["grupo"]
        grupos.append(grupo)
        filas_cache.append(
            {
                **metadatos,
                "incluida": 1,
                **dict(zip(COLUMNAS_CARACTERISTICAS, vector_audio)),
            }
        )

    if filas_sin_datos:
        print(f"Se omitieron {filas_sin_datos} audios sin características válidas.")
    if valores_no_finitos:
        print(
            f"Se encontraron {valores_no_finitos} valores no finitos; "
            "se imputarán durante el entrenamiento."
        )
    guardar_cache_caracteristicas(ruta_csv, filas_cache)
    return np.array(X), np.array(y), np.array(grupos)


def crear_modelos_experimentos():
    return {
        "RandomForest_base": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
        "RandomForest_regularizado": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
            ),
        ),
        "KNN_k5": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "KNN_k15": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15),
        ),
        "SVM_C0.1": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVC(C=0.1, random_state=42),
        ),
        "SVM_C1": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVC(C=1.0, random_state=42),
        ),
        "GMM_1_componente": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            GaussianMixtureClassifier(
                n_components=1,
                covariance_type="diag",
                random_state=42,
                max_iter=100,
            ),
        ),
        "GMM_2_componentes": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            GaussianMixtureClassifier(
                n_components=2,
                covariance_type="diag",
                random_state=42,
                max_iter=100,
            ),
        ),
    }


def evaluar_experimentos(X, y, grupos, carpeta_salida):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    if X.ndim != 2 or len(X) != len(y) or len(X) != len(grupos):
        raise ValueError("X, y y grupos deben tener el mismo número de filas.")
    if len(X) == 0:
        raise ValueError("No hay características disponibles para experimentar.")

    filas_validas = np.isfinite(X).any(axis=1)
    X = X[filas_validas].copy()
    y = y[filas_validas]
    grupos = grupos[filas_validas]
    X[~np.isfinite(X)] = np.nan
    etiquetas = np.unique(y)
    y_codificado = LabelEncoder().fit_transform(y)

    validacion = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    modelos = crear_modelos_experimentos()
    filas_resultado = []

    for nombre, modelo_base in modelos.items():
        predicciones_oof = np.full(len(y_codificado), -1, dtype=int)
        for numero_fold, (indices_train, indices_validacion) in enumerate(
            validacion.split(X, y_codificado, groups=grupos),
            start=1,
        ):
            if not np.array_equal(
                np.unique(y_codificado[indices_train]),
                np.unique(y_codificado),
            ) or not np.array_equal(
                np.unique(y_codificado[indices_validacion]),
                np.unique(y_codificado),
            ):
                raise ValueError(
                    f"El fold {numero_fold} no contiene todas las clases."
                )

            modelo = clone(modelo_base)
            modelo.fit(X[indices_train], y_codificado[indices_train])
            pred_train = modelo.predict(X[indices_train])
            pred_validacion = modelo.predict(X[indices_validacion])
            predicciones_oof[indices_validacion] = pred_validacion
            filas_resultado.append(
                {
                    "modelo": nombre,
                    "fold": numero_fold,
                    "accuracy_entrenamiento": accuracy_score(
                        y_codificado[indices_train],
                        pred_train,
                    ),
                    "accuracy_validacion": accuracy_score(
                        y_codificado[indices_validacion],
                        pred_validacion,
                    ),
                    "f1_macro_validacion": f1_score(
                        y_codificado[indices_validacion],
                        pred_validacion,
                        labels=np.arange(len(etiquetas)),
                        average="macro",
                        zero_division=0,
                    ),
                    "hablantes_entrenamiento": len(
                        np.unique(grupos[indices_train])
                    ),
                    "hablantes_validacion": len(
                        np.unique(grupos[indices_validacion])
                    ),
                }
            )
            print(
                f"{nombre}: fold {numero_fold}/5 evaluado",
                flush=True,
            )

        if np.any(predicciones_oof < 0):
            raise RuntimeError(
                f"La validación cruzada no generó predicciones para todos los "
                f"ejemplos de {nombre}."
            )

        matriz_confusion_oof = confusion_matrix(
            y_codificado,
            predicciones_oof,
            labels=np.arange(len(etiquetas)),
        )
        carpeta_matrices = (
            carpeta_salida / "salidas" / "validacion_cruzada" / nombre
        )
        carpeta_matrices.mkdir(parents=True, exist_ok=True)
        ruta_matriz_csv = carpeta_matrices / "matriz_confusion.csv"
        with ruta_matriz_csv.open("w", newline="", encoding="utf-8") as archivo_csv:
            escritor = csv.writer(archivo_csv)
            escritor.writerow(["Etiqueta real / predicha", *etiquetas])
            for etiqueta, fila in zip(etiquetas, matriz_confusion_oof):
                escritor.writerow([etiqueta, *fila])

        matriz_normalizada = confusion_matrix(
            y_codificado,
            predicciones_oof,
            labels=np.arange(len(etiquetas)),
            normalize="true",
        )
        visualizacion = ConfusionMatrixDisplay(
            confusion_matrix=matriz_normalizada,
            display_labels=etiquetas,
        )
        visualizacion.plot(values_format=".1%")
        plt.title(f"Matriz de confusión CV (out-of-fold) - {nombre}")
        plt.xlabel("Clase predicha")
        plt.ylabel("Clase real")
        ruta_matriz_png = carpeta_matrices / "matriz_confusion_porcentaje.png"
        plt.savefig(ruta_matriz_png, bbox_inches="tight")
        plt.close()

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_resultados = carpeta_salida / "experimentos_validacion_cruzada.csv"
    with ruta_resultados.open("w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=list(filas_resultado[0]),
        )
        escritor.writeheader()
        escritor.writerows(filas_resultado)

    resumen = {}
    for nombre in modelos:
        filas_modelo = [
            fila for fila in filas_resultado if fila["modelo"] == nombre
        ]
        accuracies = np.asarray(
            [fila["accuracy_validacion"] for fila in filas_modelo]
        )
        f1_scores = np.asarray(
            [fila["f1_macro_validacion"] for fila in filas_modelo]
        )
        resumen[nombre] = {
            "accuracy_media": float(np.mean(accuracies)),
            "accuracy_desv": float(np.std(accuracies)),
            "f1_macro_medio": float(np.mean(f1_scores)),
        }

    print(f"Resultados de experimentos guardados en: {ruta_resultados}")
    for nombre, metricas in resumen.items():
        print(
            f"{nombre}: accuracy={metricas['accuracy_media']:.3f} "
            f"+/- {metricas['accuracy_desv']:.3f}, "
            f"F1-macro={metricas['f1_macro_medio']:.3f}"
        )
    return resumen


def entrenar_candidatos_finales(X, y, carpetas_salida):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    filas_validas = np.isfinite(X).any(axis=1)
    X = X[filas_validas].copy()
    y = y[filas_validas]
    X[~np.isfinite(X)] = np.nan
    if len(X) == 0:
        raise ValueError("No hay ejemplos con características válidas para entrenar.")

    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y)
    carpetas = [Path(carpeta) for carpeta in carpetas_salida]
    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)
        joblib.dump(codificador, carpeta / "LabelEncoder.joblib")

    modelos = crear_modelos_experimentos()
    for nombre, modelo in modelos.items():
        modelo.fit(X, y_codificado)
        for carpeta in carpetas:
            ruta_modelo = carpeta / f"{nombre}.joblib"
            joblib.dump(modelo, ruta_modelo)
            if nombre == "RandomForest_regularizado":
                joblib.dump(
                    modelo,
                    carpeta / "RandomForestClassifier.joblib",
                )
        print(
            f"{nombre}: entrenado con {len(X)} audios y guardado en "
            + ", ".join(str(carpeta / f"{nombre}.joblib") for carpeta in carpetas),
            flush=True,
        )
        if nombre == "RandomForest_regularizado":
            print(
                "RandomForest_regularizado: publicado también como "
                "RandomForestClassifier.joblib para P1.py.",
                flush=True,
            )


def entrenar_modelo_audio(X, y, carpeta_salida, grupos=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    filas_validas = np.isfinite(X).any(axis=1)
    filas_omitidas = int((~filas_validas).sum())
    if filas_omitidas:
        print(
            f"Se omitieron {filas_omitidas} ejemplos sin características válidas."
        )
    X = X[filas_validas].copy()
    y = y[filas_validas]
    if grupos is None:
        raise ValueError(
            "Se requieren los identificadores de hablante para evitar "
            "que sus segmentos queden repartidos entre entrenamiento y prueba."
        )
    grupos = np.asarray(grupos)
    if len(grupos) != len(filas_validas):
        raise ValueError("Debe proporcionarse un identificador de hablante por audio.")
    grupos = grupos[filas_validas]
    X[~np.isfinite(X)] = np.nan
    if len(X) == 0:
        raise ValueError("No hay ejemplos con características válidas para entrenar.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    particionador = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    indices_entrenamiento, indices_prueba = next(
        particionador.split(X, y_encoded, groups=grupos)
    )
    X_train, X_test = X[indices_entrenamiento], X[indices_prueba]
    y_train, y_test = y_encoded[indices_entrenamiento], y_encoded[indices_prueba]
    if not np.array_equal(np.unique(y_train), np.unique(y_encoded)):
        raise ValueError(
            "La división por hablante dejó una clase fuera del conjunto de entrenamiento."
        )
    if not np.array_equal(np.unique(y_test), np.unique(y_encoded)):
        raise ValueError(
            "La división por hablante dejó una clase fuera del conjunto de prueba."
        )

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_encoder = carpeta_salida / "LabelEncoder.joblib"
    joblib.dump(le, ruta_encoder)
    print(f"Codificador de etiquetas guardado en: {ruta_encoder}")

    clasificadores = {

        "RandomForestClassifier": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
        "KNeighborsClassifier_n5": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "SVM_SVC": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVC(random_state=42),
        ),
        "GaussianMixtureClassifier": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            GaussianMixtureClassifier(
                n_components=2,
                covariance_type="diag",
                random_state=42,
                max_iter=100,
            ),
        ),
    }

    for nombre, modelo in clasificadores.items():
        print(f"\nEntrenando el modelo {nombre}...")
        modelo.fit(X_train, y_train)

        ruta_modelo = carpeta_salida / f"{nombre}.joblib"
        joblib.dump(modelo, ruta_modelo)
        print(f"Modelo guardado en: {ruta_modelo}")

        y_pred = modelo.predict(X_test)
        exactitud = accuracy_score(y_test, y_pred)
        reporte = classification_report(
            y_test, y_pred, target_names=le.classes_
        )

        print(f"\nExactitud (Accuracy): {exactitud * 100:.2f}%\n")
        print("Reporte de clasificación detallado:")
        print(reporte)

        matriz_confusion = confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(len(le.classes_)),
        )
        carpeta_salidas = carpeta_salida / "salidas" / nombre
        carpeta_salidas.mkdir(parents=True, exist_ok=True)

        ruta_metricas = carpeta_salidas / "metricas.txt"
        ruta_metricas.write_text(
            f"Modelo: {nombre}\n"
            "Evaluación: holdout de un único pliegue; no es la matriz agregada "
            "de validación cruzada.\n"
            f"Exactitud (Accuracy): {exactitud * 100:.2f}%\n\n"
            f"Reporte de clasificación:\n{reporte}",
            encoding="utf-8",
        )
        print(f"Métricas guardadas en: {ruta_metricas}")

        ruta_matriz_csv = carpeta_salidas / "matriz_confusion.csv"
        with ruta_matriz_csv.open(
            "w", newline="", encoding="utf-8"
        ) as archivo_csv:
            escritor = csv.writer(archivo_csv)
            escritor.writerow(["Etiqueta real / predicha", *le.classes_])
            for etiqueta, fila in zip(le.classes_, matriz_confusion):
                escritor.writerow([etiqueta, *fila])
        print(f"Matriz de confusión guardada en: {ruta_matriz_csv}")

        visualizacion = ConfusionMatrixDisplay(
            confusion_matrix=matriz_confusion,
            display_labels=le.classes_,
        )
        visualizacion.plot(values_format="d")
        plt.title(f"Matriz de confusión holdout (un pliegue) - {nombre}")
        plt.xlabel("Hablante identificado")
        plt.ylabel("Hablante real")

        ruta_matriz_png = carpeta_salidas / "matriz_confusion.png"
        plt.savefig(ruta_matriz_png, bbox_inches="tight")
        print(f"Imagen de la matriz guardada en: {ruta_matriz_png}")
        plt.close()

    return clasificadores, le

def main(carpeta_output, solo_experimentos=False, guardar_candidatos=False):
    # Carpeta donde tienes tus audios listos
    carpeta_dataset = "Dataset_procesado_10"
    # hay también "Dataset_procesado_5",
    # "Dataset_procesado_10"
    # "Dataset_procesado_15"
    
    # 1. Crear matriz de características y etiquetas
    X, y, grupos = crear_matriz_caracteristicas(carpeta_dataset)
    
    if X is not None and len(X) > 0:
        if guardar_candidatos:
            evaluar_experimentos(X, y, grupos, carpeta_output)
            entrenar_candidatos_finales(
                X,
                y,
                [carpeta_output, Path(__file__).resolve().parent / "Modelo"],
            )
            return
        if solo_experimentos:
            evaluar_experimentos(X, y, grupos, carpeta_output)
            return
        # 2. Entrenar el modelo
        entrenar_modelo_audio(X, y, carpeta_output, grupos=grupos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experimentos",
        action="store_true",
        help=(
            "Compara configuraciones con validación cruzada por hablante; "
            "reutiliza caracteristicas.csv si existe."
        ),
    )
    parser.add_argument(
        "--guardar-modelos-cv",
        action="store_true",
        help=(
            "Ejecuta validación cruzada por hablante y luego reentrena las "
            "configuraciones evaluadas con todos los datos; guarda los modelos "
            "en output/ y Modelo/."
        ),
    )
    argumentos = parser.parse_args()
    carpeta_output = Path(__file__).resolve().parent / "output"
    carpeta_output.mkdir(parents=True, exist_ok=True)
    ruta_terminal = carpeta_output / "salidas" / "terminal.txt"
    ruta_terminal.parent.mkdir(parents=True, exist_ok=True)

    with ruta_terminal.open("w", encoding="utf-8") as archivo_terminal:
        salida = SalidaDuplicada(sys.stdout, archivo_terminal)
        errores = SalidaDuplicada(sys.stderr, archivo_terminal)
        with redirect_stdout(salida), redirect_stderr(errores):
            try:
                if argumentos.experimentos and argumentos.guardar_modelos_cv:
                    parser.error(
                        "--experimentos y --guardar-modelos-cv son opciones excluyentes."
                    )
                main(
                    carpeta_output,
                    solo_experimentos=argumentos.experimentos,
                    guardar_candidatos=argumentos.guardar_modelos_cv,
                )
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)