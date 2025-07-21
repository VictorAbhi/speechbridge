from flask import Flask, request, render_template
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import subprocess
import tempfile

app = Flask(__name__)

# Setup ASR model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "distil-whisper/distil-small.en"

processor = AutoProcessor.from_pretrained(model_id)
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1
)

# Setup Translation model
translation_model_name = "facebook/nllb-200-distilled-600M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(device)

def translate_text_nllb(input_text, src_lang="eng_Latn", tgt_lang="npi_Deva"):
    translation_tokenizer.src_lang = src_lang
    inputs = translation_tokenizer(input_text, return_tensors="pt").to(device)

    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=256
    )
    translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text



@app.route("/", methods=["GET", "POST"])
def transcribe():
    global transcript
    transcript = ""
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file selected", 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            file.save(temp_input.name)
            audio_path = temp_input.name.replace(".mp4", "_audio.wav")

        ffmpeg_cmd = ["ffmpeg", "-i", temp_input.name, "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        result = asr_pipe(audio_path)
        transcript = result.get("text", "")

        os.remove(temp_input.name)
        os.remove(audio_path)

    return render_template("index.html", transcript=transcript)


@app.route('/translate', methods=['POST'])
def translate_text():
    if not transcript:
        return "first transcribe audio or video file", 400

    translation = translate_text_nllb(transcript)
    return render_template('index.html', translation=translation)


if __name__ == "__main__":
    app.run(debug=True)
