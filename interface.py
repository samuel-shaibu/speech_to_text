import whisper
import gradio as gr
import numpy as np
import tempfile
import scipy.io.wavfile
import datetime
import os

# Load Whisper model
model = whisper.load_model("base")

# Transcript file
TRANSCRIPT_FILE = "transcript.txt"

def transcribe_file(file_path):
    """Transcribe an audio file and save the result."""
    try:
        if file_path is None:
            return "No audio received."

        print("🧠 Transcribing...")
        result = model.transcribe(file_path)
        text = result["text"].strip()

        # Save to transcript
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"{timestamp} → {text}"
        print(f"📝 {log}")

        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(log + "\n")

        return log

    except Exception as e:
        return f"❌ Error: {e}"

# Gradio Interface
interface = gr.Interface(
    fn=transcribe_file,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak into your mic"),
    outputs=gr.Textbox(label="📝 Transcription"),
    title="Whisper Speech-to-Text",
    description="Speak clearly and Whisper will transcribe your voice. Output is saved to transcript.txt."
)

if _name_ == "_main_":
    print("🚀 Launching Gradio App...")
    interface.launch()
