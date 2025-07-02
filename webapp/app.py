from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import os
import subprocess
import tempfile

app = Flask(__name__)

# Setup model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "distil-whisper/distil-small.en"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device
)

@app.route("/", methods=["GET", "POST"])
def transcribe():
    transcript = ""
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            file.save(temp_input.name)
            audio_path = temp_input.name.replace(".mp4", "_audio.wav")

        # Extract audio using ffmpeg (resampled to 16kHz mono)
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", temp_input.name,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            audio_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transcribe
        result = pipe(audio_path)
        transcript = result["text"]

        # Clean up temp files
        os.remove(temp_input.name)
        os.remove(audio_path)

    return render_template("index.html", transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True)
