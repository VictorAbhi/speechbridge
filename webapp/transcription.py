import torch
from transformers import pipeline
import os
import librosa
from tempfile import mkdtemp
import soundfile as sf

class TranscriptionService:
    def __init__(self, model_name="openai/whisper-small"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=self.device)

    def transcribe(self, file_path, chunk_size_s=30):
        """Handles large audio files by chunking"""

        audio, sr = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)
        temp_dir = mkdtemp()
        transcription = ""

        for start in range(0, int(duration), chunk_size_s):
            end = min(start + chunk_size_s, duration)
            chunk = audio[start * sr:end * sr]
            chunk_path = os.path.join(temp_dir, f"chunk_{start}.wav")
            sf.write(chunk_path, chunk, sr)

            result = self.pipe(chunk_path)
            transcription += result['text'] + " "

        return transcription.strip()
