import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import time
import threading
import queue
import scipy.io.wavfile
import datetime

# Load Whisper model
model = whisper.load_model("base")

SAMPLE_RATE = 16000
DURATION = 10
CHANNELS = 1
TRANSCRIPT_FILE = "transcript.txt"
audio_queue = queue.Queue()

def record_audio():
    while True:
        print("\n🟢 Now! Speak clearly for the next", DURATION, "seconds...")
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        audio_queue.put(audio.copy())

def transcribe():
    while True:
        if not audio_queue.empty():
            audio = audio_queue.get()

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    audio_int16 = np.int16(audio * 32767)
                    scipy.io.wavfile.write(temp_path, SAMPLE_RATE, audio_int16)

                print("🧠 Transcribing...")
                result = model.transcribe(temp_path)
                text = result["text"].strip()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"📝 {timestamp} → You said: {text}")

                with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} → {text}\n")

                os.remove(temp_path)
            except Exception as e:
                print(f"❌ Error: {e}")

threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=transcribe, daemon=True).start()

while True:
    time.sleep(1)
