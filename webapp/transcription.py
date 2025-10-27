import torch
import numpy as np
import librosa
import soundfile as sf
import os
import shutil
from tempfile import mkdtemp
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

class TranscriptionService:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialize the transcription service with Whisper model
        Using the same approach as your working notebook
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model (similar to your notebook)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            self.model.to(self.device)
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to pipeline approach
            from transformers import pipeline
            self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=0 if torch.cuda.is_available() else -1)
            self.model_loaded = False

    def load_audio_chunks(self, file_path, chunk_duration=30, sample_rate=16000, overlap=1.5):
        """
        Load audio and split into overlapping chunks
        Based on your notebook implementation
        """
        try:
            # Load audio with librosa (converts to mono and resamples)
            audio, sr = librosa.load(file_path, sr=sample_rate)
            
            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            chunk_samples = int(chunk_duration * sample_rate)
            step = int((chunk_duration - overlap) * sample_rate)

            chunks = []
            for start in range(0, len(audio), step):
                end = start + chunk_samples
                if end > len(audio):
                    # Pad the last chunk if it's shorter
                    chunk = np.pad(audio[start:], (0, max(0, end - len(audio))))
                else:
                    chunk = audio[start:end]
                
                # Only add chunks that have meaningful audio (not just silence)
                if np.max(np.abs(chunk)) > 0.001:  # Threshold for silence
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error loading audio chunks: {str(e)}")

    def transcribe_chunks(self, chunks, sample_rate=16000):
        """
        Transcribe audio chunks using the manual approach from your notebook
        """
        if not self.model_loaded:
            raise Exception("Model not properly loaded")
            
        full_transcript = ""
        
        for i, chunk in enumerate(chunks):
            try:
                # Process chunk through the processor (same as your notebook)
                inputs = self.processor(
                    chunk, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate transcription
                with torch.no_grad():  # Save memory
                    predicted_ids = self.model.generate(inputs)
                
                # Decode the result
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Clean up the transcription
                transcription = transcription.strip()
                if transcription and len(transcription) > 1:  # Avoid single character results
                    full_transcript += transcription + " "
                    print(f"Chunk {i+1}/{len(chunks)}: {transcription[:50]}...")
                else:
                    print(f"Chunk {i+1}/{len(chunks)}: No meaningful speech detected")
                    
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")
                continue
        
        result = full_transcript.strip()
        return result if result else "No clear speech detected in the audio file."

    def transcribe(self, file_path, chunk_size_s=30):
        """
        Main transcription method using the chunking approach from your notebook
        """
        try:
            if self.model_loaded:
                # Use the advanced method (same as your notebook)
                chunks = self.load_audio_chunks(file_path, chunk_duration=chunk_size_s, overlap=1.5)
                
                if not chunks:
                    return "No audio content detected in the file."
                
                print(f"🎵 Processing {len(chunks)} audio chunks...")
                return self.transcribe_chunks(chunks)
            else:
                # Fallback to pipeline method
                return self.transcribe_simple(file_path)
                
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

    def transcribe_simple(self, file_path):
        """
        Simple transcription without chunking (fallback method)
        """
        try:
            if hasattr(self, 'pipe'):
                result = self.pipe(file_path)
                return result['text'].strip() if result and 'text' in result else "No speech detected."
            else:
                raise Exception("No transcription method available")
        except Exception as e:
            raise Exception(f"Simple transcription error: {str(e)}")

    def get_audio_info(self, file_path):
        """
        Get basic audio file information
        """
        try:
            audio, sr = librosa.load(file_path, sr=None)  # Keep original sample rate
            duration = len(audio) / sr
            
            # Check if audio has meaningful content
            max_amplitude = np.max(np.abs(audio))
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'channels': 1,  # librosa loads as mono by default
                'max_amplitude': float(max_amplitude),
                'has_audio': max_amplitude > 0.001
            }
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")

    def preprocess_audio(self, input_path, output_path=None):
        """
        Optional: Preprocess audio to improve transcription quality
        (Similar to your denoising step in the notebook)
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=16000)  # Standard rate for Whisper
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Save processed audio if output path provided
            if output_path:
                sf.write(output_path, audio, sr)
                return output_path
            else:
                # Create temporary file
                temp_dir = mkdtemp()
                temp_path = os.path.join(temp_dir, "processed_audio.wav")
                sf.write(temp_path, audio, sr)
                return temp_path
                
        except Exception as e:
            raise Exception(f"Audio preprocessing error: {str(e)}")