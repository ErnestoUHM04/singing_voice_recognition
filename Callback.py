import sounddevice as sd
import soundfile as sf
import numpy as np
import threading

# ==========================================
# CLASE PARA REPRODUCIR AUDIO
# ==========================================

class ReproductorAudio:

    def __init__(self, archivo, device=None):

        self.audio, self.frecuencia = sf.read(
            archivo,
            dtype="float32"
        )

        # Convertir audio mono a formato de dos dimensiones-
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(-1, 1)

        self.device = device
        self.posicion = 0
        self.reproduciendo = False
        self.pausado = False

        self.stream = None
        self.lock = threading.Lock()

    def callback(self, outdata, frames, time, status):

        if status:
            print("Estado de reproducción:", status)

        with self.lock:

            restante = len(self.audio) - self.posicion
            cantidad = min(frames, restante)

            if cantidad > 0:
                outdata[:cantidad] = self.audio[
                    self.posicion:self.posicion + cantidad
                ]

                self.posicion += cantidad

            if cantidad < frames:
                outdata[cantidad:] = 0
                self.reproduciendo = False
                raise sd.CallbackStop

    def reproducir(self):

        with self.lock:

            if self.reproduciendo:
                return

            if self.posicion >= len(self.audio):
                self.posicion = 0

            self.reproduciendo = True
            self.pausado = False

        self.stream = sd.OutputStream(
            samplerate=self.frecuencia,
            channels=self.audio.shape[1],
            dtype="float32",
            device=self.device,
            callback=self.callback
        )

        self.stream.start()

    def pausar(self):

        with self.lock:

            if not self.reproduciendo:
                return

            self.reproduciendo = False
            self.pausado = True

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def reanudar(self):

        if self.pausado:
            self.reproducir()

    def detener(self):

        with self.lock:
            self.reproduciendo = False
            self.pausado = False
            self.posicion = 0

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ==========================================
# CLASE PARA GRABAR AUDIO
# ==========================================

class GrabadorAudio:

    def __init__(self, device=None, frecuencia=44100, canales=1):

        self.device = device
        self.frecuencia = frecuencia
        self.canales = canales

        self.grabando = False
        self.stream = None

        self.bloques_audio = []
        self.lock = threading.Lock()

    def callback(self, indata, frames, time, status):

        if status:
            print("Estado de grabación:", status)

        with self.lock:

            if self.grabando:
                self.bloques_audio.append(indata.copy())

    def iniciar_grabacion(self):

        with self.lock:

            if self.grabando:
                return

            self.bloques_audio = []
            self.grabando = True

        self.stream = sd.InputStream(
            samplerate=self.frecuencia,
            channels=self.canales,
            dtype="float32",
            device=self.device,
            callback=self.callback
        )

        self.stream.start()

        print("Grabación iniciada")

    def detener_grabacion(self):

        with self.lock:

            if not self.grabando:
                return None

            self.grabando = False

        self.stream.stop()
        self.stream.close()
        self.stream = None

        with self.lock:

            if not self.bloques_audio:
                return np.array([], dtype=np.float32)

            audio = np.concatenate(
                self.bloques_audio,
                axis=0
            )

        print("Grabación detenida")

        return audio