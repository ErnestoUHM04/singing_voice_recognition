from pathlib import Path
from pydub import AudioSegment


def segmentos_de_audio(carpeta_entrada, carpeta_salida, duracion=5, solapamiento=0.5):
    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    duracion_ms = duracion * 1000
    # Si el solapamiento es 0.5 (50%), el salto será de 2.5 segundos (2500 ms)
    salto_ms = int(duracion_ms * (1 - solapamiento))

    for archivo in carpeta_entrada.glob("*.wav"):
        try:
            audio = AudioSegment.from_wav(archivo)
        except Exception as e:
            print(f"Error al leer {archivo.name}: {e}")
            continue

        # Validar que el archivo sea más largo que el segmento mínimo
        if len(audio) < duracion_ms:
            print(f"Aviso: {archivo.name} es muy corto (menos de {duracion}s), se omite.")
            continue

        # Generar segmentos con ventana deslizante
        for numero, inicio in enumerate(
            range(0, len(audio) - duracion_ms + 1, salto_ms),
            start=1,
        ):
            segmento = audio[inicio : inicio + duracion_ms]
            nombre = f"{archivo.stem}_seg_{numero:03d}.wav"

            segmento.export(
                carpeta_salida / nombre,
                format="wav",
            )

    print(f"¡Proceso finalizado! Segmentos guardados en: {carpeta_salida}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        inp = sys.argv[1]
        out = sys.argv[2]
    else:
        # ¡Aquí solo pones el nombre de la carpeta si está al lado del script!
        print("Usando carpetas locales...")
        inp = "Dataset_procesado"# HAY QUE CAMBIARLO A DATASET_PROCESADO
        out = "Dataset_segmentos"

    segmentos_de_audio(inp, out, duracion=5, solapamiento=0.5)