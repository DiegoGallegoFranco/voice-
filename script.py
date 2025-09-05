import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time as _time

# Diarización (opcional con pyannote)
DIARIZATION_AVAILABLE = False
try:
    from pyannote.audio import Pipeline  # type: ignore
    DIARIZATION_AVAILABLE = True
except Exception:
    DIARIZATION_AVAILABLE = False

# Cargar el modelo Whisper
model = whisper.load_model("base")  # Puedes usar "small" o "medium" para mayor precisión

# Configuración de audio
samplerate = 16000  # Whisper requiere 16kHz
block_duration = 5  # segundos por bloque
block_size = int(samplerate * block_duration)
audio_queue = queue.Queue()

# Buffer para ventanas más largas (para diarización)
window_seconds = 15  # tamaño de ventana a procesar
stride_seconds = 10   # cada cuánto procesar
window_size = int(samplerate * window_seconds)
stride_size = int(samplerate * stride_seconds)

buffer_lock = threading.Lock()
audio_buffer = np.zeros(0, dtype=np.float32)
start_time_ref = _time.time()
processed_until = 0  # índice de muestra ya procesado

# Inicializar pipeline de diarización si disponible y token presente
_diaria_pipeline = None
if DIARIZATION_AVAILABLE:
    import os
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            _diaria_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("✅ Diarización habilitada con pyannote.audio.")
        except Exception as e:
            print(f"⚠️ No se pudo inicializar la diarización: {e}. Continuando sin diarización…")
            _diaria_pipeline = None
    else:
        print("ℹ️ Para habilitar diarización, define la variable de entorno HUGGINGFACE_TOKEN con tu token de Hugging Face.")
else:
    print("ℹ️ pyannote.audio no está instalado. Se continuará sin diarización.")

# Captura de audio en tiempo real
def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ Error de audio: {status}")
    mono_audio = np.mean(indata, axis=1).astype(np.float32)  # Convertir a mono si es necesario
    with buffer_lock:
        global audio_buffer
        audio_buffer = np.concatenate([audio_buffer, mono_audio])
    audio_queue.put(len(mono_audio))  # señal para el hilo de procesamiento


def _transcribe_segment(segment_audio: np.ndarray) -> str:
    segment_audio = whisper.pad_or_trim(segment_audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)
    options = whisper.DecodingOptions(language="es", task="transcribe", fp16=False)
    result = model.decode(mel, options)
    return result.text.strip()

# Hilo de transcripción con diarización (si disponible)
def transcribe_loop():
    print("🎧 Escuchando en español... presiona Ctrl+C para salir.")
    global processed_until
    try:
        while True:
            # Espera a que llegue algo nuevo
            _ = audio_queue.get()
            with buffer_lock:
                total_len = len(audio_buffer)

            # Procesar por ventanas deslizantes para baja latencia
            while True:
                with buffer_lock:
                    available = len(audio_buffer) - processed_until
                if available < stride_size:
                    break

                # Determinar rango de ventana a procesar
                start_idx = max(0, processed_until - (window_size - stride_size))
                end_idx = start_idx + window_size
                with buffer_lock:
                    window = audio_buffer[start_idx:min(end_idx, len(audio_buffer))].copy()
                window_offset = start_idx  # para mapear a índices globales

                if _diaria_pipeline is not None and len(window) > 0:
                    try:
                        # Ejecutar diarización sobre la ventana completa
                        # pyannote espera 16kHz mono
                        diarization = _diaria_pipeline({"waveform": np.expand_dims(window, 0), "sample_rate": samplerate})
                        # Para cada turno de habla, transcribir ese trozo
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            seg_start = int(turn.start * samplerate)
                            seg_end = int(turn.end * samplerate)
                            seg_start = max(0, seg_start)
                            seg_end = min(len(window), seg_end)
                            if seg_end - seg_start <= int(0.2 * samplerate):
                                continue  # saltar segmentos muy cortos
                            segment_audio = window[seg_start:seg_end]
                            texto = _transcribe_segment(segment_audio)
                            if texto:
                                print(f"👤 {speaker}: {texto}")
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"⚠️ Error en diarización: {e}. Transcribiendo sin etiquetas…")
                        texto = _transcribe_segment(window)
                        if texto:
                            print(f"📝 Texto: {texto}")
                else:
                    # Sin diarización: transcribir la ventana
                    texto = _transcribe_segment(window)
                    if texto:
                        print(f"📝 Texto: {texto}")

                processed_until = start_idx + stride_size
    except KeyboardInterrupt:
        print("\n🛑 Transcripción detenida por el usuario.")

# Iniciar grabación y transcripción
with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, blocksize=block_size):
    threading.Thread(target=transcribe_loop, daemon=True).start()
    input("Presiona Enter para detener...\n")
