import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment

def normalizar_audio(audio):
    amplitud_maxima = np.max(np.abs(audio))
    if amplitud_maxima == 0:
        return audio
    return audio / amplitud_maxima

def eliminar_silencios_globales(audio, top_db=30):
    intervalos_voz = librosa.effects.split(audio, top_db=top_db)
    segmentos_voz = []
    for inicio, fin in intervalos_voz:
        segmentos_voz.append(audio[inicio:fin])
    
    if len(segmentos_voz) == 0:
        return audio # Si no detectó nada, devuelve el original
    
    return np.concatenate(segmentos_voz)

def procesar_y_segmentar(carpeta_entrada, carpeta_salida, duracion, solapamiento):
    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    duracion_ms = duracion * 1000
    salto_ms = int(duracion_ms * (1 - solapamiento))

    contador_total = 0

    for archivo in carpeta_entrada.glob("*.wav"):
        print(f"Procesando: {archivo.name}")
        
        try:
            # 1. Cargar audio con librosa
            audio_np, sr = librosa.load(archivo, sr=None, mono=True)
            
            # 2. Limpiar silencios globales para que el audio sea continuo
            audio_limpio = eliminar_silencios_globales(audio_np, top_db=30)
            
            # 3. Guardar temporalmente el audio limpio
            ruta_temporal = carpeta_salida / f"temp_{archivo.name}"
            sf.write(ruta_temporal, audio_limpio, sr)
            
            # 4. Cargar con pydub y FORZAR explícitamente a MONO (.set_channels(1))
            audio_pydub = AudioSegment.from_wav(ruta_temporal).set_channels(1)
            
            # Borrar archivo temporal
            ruta_temporal.unlink(missing_ok=True)

            # Validar duración mínima
            if len(audio_pydub) < duracion_ms:
                print(f"  [Aviso] {archivo.name} es muy corto (menos de {duracion}s), se omite.")
                continue

            # 5. Generar segmentos con ventana deslizante y solapamiento
            segmentos_creados = 0
            for numero, inicio in enumerate(
                range(0, len(audio_pydub) - duracion_ms + 1, salto_ms),
                start=1,
            ):
                segmento = audio_pydub[inicio : inicio + duracion_ms]
                
                # Convertir de nuevo a numpy array para verificar volumen real
                samples = np.array(segmento.get_array_of_samples(), dtype=np.float32) / 32768.0
                
                # Si el segmento es prácticamente silencio, lo saltamos
                if np.max(np.abs(samples)) < 0.02:
                    continue
                
                # Normalizar el segmento limpio
                segmento_normalizado = normalizar_audio(samples)
                
                # Nombre y ruta final
                nombre = f"{archivo.stem}_seg_{numero:03d}.wav"
                ruta_final = carpeta_salida / nombre
                
                # Guardar resultado final
                sf.write(ruta_final, segmento_normalizado, sr)
                segmentos_creados += 1
                contador_total += 1

            print(f"  -> Generados {segmentos_creados} segmentos en Mono, limpios y normalizados.\n" + "-"*40)

        except Exception as e:
            print(f"  [Error] Al procesar {archivo.name}: {e}\n" + "-"*40)

    print(f"¡Proceso finalizado! Total de segmentos guardados: {contador_total} en '{carpeta_salida}'")


if __name__ == "__main__":
    inp = "Dataset_limpio"
    out = "Dataset_procesado_10_s_25"
    
    # Ejecutamos con 5 segundos de duración y 50% de solapamiento
    procesar_y_segmentar(inp, out, duracion=10, solapamiento=0.25)