import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models once
@st.cache_resource
def load_models():
    asr_model = whisper.load_model("small")

    translation_model_name = "d2niraj555/mt5-eng2nep"
    trans_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

    return asr_model, trans_tokenizer, trans_model

asr_model, trans_tokenizer, trans_model = load_models()

# UI
st.title("🎙️ SpeechBridge")
st.subheader("Transcribe → Translate → Summarize")

uploaded_file = st.file_uploader("Upload an English audio file (.mp3 or .wav)", type=["mp3", "wav"])

import tempfile

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_path = temp_audio_file.name

    with st.spinner("Transcribing..."):
        transcription = asr_model.transcribe(temp_audio_path)["text"]
        st.success("Transcription complete.")
        st.text_area("📄 Transcription (English)", transcription, height=150)

    with st.spinner("Translating to Nepali..."):
        input_text = f"translate English to Nepali: {transcription}"
        inputs = trans_tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = trans_model.generate(**inputs, max_length=256)
        nepali_translation = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("Translation complete.")
        st.text_area("🗣️ Nepali Translation", nepali_translation, height=150)

    st.markdown("---")