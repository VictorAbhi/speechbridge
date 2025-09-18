import torch
import numpy as np
import librosa
import soundfile as sf
import os
import shutil
from tempfile import mkdtemp
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

class TranscriptionService:
    def __init__(self):  
        """
        Initialize the transcription service with multiple models for English and Nepali
        Based on your notebook recommendations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Models for different languages
        self.models = {}
        self.processors = {}
        self.pipelines = {}
        
        # Language model mappings based on your notebook
        self.language_models = {
            'en': 'BlueRaccoon/whisper-small-en',  # English model from notebook
            'ne': 'amitpant7/Nepali-Automatic-Speech-Recognition'  # Nepali model from notebook
        }
        
        # Initialize models lazily (load when needed)
        self.initialized_languages = set()

    def _initialize_language_model(self, language):
        """Initialize model for specific language"""
        if language in self.initialized_languages:
            return
            
        model_name = self.language_models.get(language, self.language_models['en'])
        
        try:
            print(f"Loading {language} model: {model_name}")
            
            # Try loading processor and model directly
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            model.to(self.device)
            
            self.processors[language] = processor
            self.models[language] = model
            
            print(f"✅ {language} model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading {language} model directly: {e}")
            # Fallback to pipeline approach
            try:
                pipe = pipeline(
                    "automatic-speech-recognition", 
                    model=model_name, 
                    device=0 if torch.cuda.is_available() else -1,
                    return_timestamps=True
                )
                self.pipelines[language] = pipe
                print(f"✅ {language} pipeline loaded successfully")
                
            except Exception as pipe_error:
                print(f"❌ Failed to load {language} pipeline: {pipe_error}")
                # Use English as fallback for any language
                if language != 'en':
                    print(f"Using English model as fallback for {language}")
                    self._initialize_language_model('en')
                    return
                else:
                    raise Exception("Failed to load any transcription model")
        
        self.initialized_languages.add(language)

    def load_audio_chunks(self, file_path, chunk_duration=30, sample_rate=16000, overlap=1.5):
        """
        Load audio and split into overlapping chunks
        Based on your notebook implementation with denoising
        """
        try:
            # Load audio with librosa (converts to mono and resamples)
            audio, sr = librosa.load(file_path, sr=sample_rate)

            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")

            # Apply basic denoising (similar to your notebook)
            audio = self._denoise_audio(audio)

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

    def _denoise_audio(self, audio, sr=16000):
        """
        Basic audio denoising similar to your notebook approach
        """
        try:
            # Simple noise reduction using spectral subtraction approach
            # Estimate noise from first 0.5 seconds
            noise_sample_length = min(int(0.5 * sr), len(audio) // 4)
            if noise_sample_length > 0:
                noise_profile = audio[:noise_sample_length]
                noise_level = np.std(noise_profile) * 2
                
                # Apply simple noise gate
                audio = np.where(np.abs(audio) > noise_level, audio, audio * 0.1)
            
            return audio
            
        except Exception as e:
            print(f"Warning: Denoising failed: {e}")
            return audio  # Return original audio if denoising fails

    def transcribe_chunks(self, chunks, sample_rate=16000, language='en'):
        """
        Transcribe audio chunks using the manual approach from your notebook
        """
        # Initialize language model if needed
        self._initialize_language_model(language)
        
        if language not in self.models and language not in self.pipelines:
            raise Exception(f"Model not properly loaded for language: {language}")

        full_transcript = ""

        for i, chunk in enumerate(chunks):
            try:
                if language in self.models:
                    # Use direct model approach (preferred)
                    processor = self.processors[language]
                    model = self.models[language]
                    
                    # Process chunk through the processor (same as your notebook)
                    inputs = processor(
                        chunk,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).input_features.to(self.device)

                    # Generate transcription
                    with torch.no_grad():  # Save memory
                        predicted_ids = model.generate(
                            inputs,
                            forced_decoder_ids=processor.get_decoder_prompt_ids(
                                language=language, 
                                task="transcribe"
                            )
                        )

                    # Decode the result
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    
                else:
                    # Use pipeline approach
                    pipe = self.pipelines[language]
                    
                    # Create temporary file for chunk
                    temp_path = f"/tmp/chunk_{i}.wav"
                    sf.write(temp_path, chunk, sample_rate)
                    
                    result = pipe(temp_path)
                    transcription = result['text'] if isinstance(result, dict) else str(result)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

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

    def transcribe(self, file_path, chunk_size_s=30, language='en'):
        """
        Main transcription method using the chunking approach from your notebook
        """
        try:
            chunks = self.load_audio_chunks(file_path, chunk_duration=chunk_size_s, overlap=1.5)

            if not chunks:
                return "No audio content detected in the file."

            print(f"🎵 Processing {len(chunks)} audio chunks for {language}...")
            return self.transcribe_chunks(chunks, language=language)

        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

    def transcribe_simple(self, file_path, language='en'):
        """
        Simple transcription without chunking (fallback method)
        """
        try:
            # Initialize language model if needed
            self._initialize_language_model(language)
            
            if language in self.pipelines:
                pipe = self.pipelines[language]
                result = pipe(file_path)
                text = result['text'] if isinstance(result, dict) else str(result)
                return text.strip() if text else "No speech detected."
                
            elif language in self.models:
                # Load audio for direct model processing
                audio, sr = librosa.load(file_path, sr=16000)
                audio = self._denoise_audio(audio)
                
                processor = self.processors[language]
                model = self.models[language]
                
                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.device)

                with torch.no_grad():
                    predicted_ids = model.generate(
                        inputs,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(
                            language=language, 
                            task="transcribe"
                        )
                    )

                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return transcription.strip() if transcription else "No speech detected."
            else:
                raise Exception(f"No transcription method available for language: {language}")
                
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
        Preprocess audio to improve transcription quality
        Includes denoising similar to your notebook approach
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=16000)  # Standard rate for Whisper

            # Apply denoising
            audio = self._denoise_audio(audio, sr)

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