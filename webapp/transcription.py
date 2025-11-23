import torch
import numpy as np
import librosa
import soundfile as sf
import os
import shutil
from tempfile import mkdtemp
import scipy.signal as sps
import noisereduce as nr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class TranscriptionService:
    def __init__(self, model_name="openai/whisper-small", language="en", max_workers=4):
        """
        Initialize the transcription service with Whisper model
        
        Args:
            model_name: Model to use for transcription
            language: Language code ('en' for English, 'ne' for Nepali)
            max_workers: Number of parallel threads for processing chunks
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.max_workers = max_workers
        self.print_lock = Lock()  # For thread-safe printing
        
        # Set model based on language
        if language == "ne":
            self.model_name = "Faith-nchifor/whisper-small-nep"
        else:
            self.model_name = model_name
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device} for language: {language}")
        except Exception as e:
            print(f"Error loading model: {e}")
            from transformers import pipeline
            self.pipe = pipeline("automatic-speech-recognition", model=self.model_name, 
                               device=0 if torch.cuda.is_available() else -1)
            self.model_loaded = False

    def load_audio_chunks(self, file_path, chunk_duration=30, sample_rate=16000, overlap=1.5):
        """
        Load audio and split into overlapping chunks
        """
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)
            
            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            chunk_samples = int(chunk_duration * sample_rate)
            step = int((chunk_duration - overlap) * sample_rate)

            chunks = []
            for start in range(0, len(audio), step):
                end = start + chunk_samples
                if end > len(audio):
                    chunk = np.pad(audio[start:], (0, max(0, end - len(audio))))
                else:
                    chunk = audio[start:end]
                
                if np.max(np.abs(chunk)) > 0.001:
                    chunks.append((len(chunks), chunk))  # Store index with chunk
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error loading audio chunks: {str(e)}")

    def transcribe_single_chunk(self, chunk_data, sample_rate=16000):
        """
        Transcribe a single audio chunk (thread-safe)
        
        Args:
            chunk_data: Tuple of (chunk_index, audio_chunk)
        
        Returns:
            Tuple of (chunk_index, transcription_text)
        """
        chunk_idx, chunk = chunk_data
        
        try:
            # Process chunk through the processor
            inputs = self.processor(
                chunk, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs)
            
            # Decode the result
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription = transcription.strip()
            
            # Thread-safe printing
            with self.print_lock:
                if transcription and len(transcription) > 1:
                    print(f"✓ Chunk {chunk_idx + 1}: {transcription[:50]}...")
                else:
                    print(f"○ Chunk {chunk_idx + 1}: No meaningful speech detected")
            
            return (chunk_idx, transcription if len(transcription) > 1 else "")
                    
        except Exception as e:
            with self.print_lock:
                print(f"❌ Error processing chunk {chunk_idx + 1}: {e}")
            return (chunk_idx, "")

    def transcribe_chunks_parallel(self, chunks, sample_rate=16000):
        """
        Transcribe audio chunks in parallel using ThreadPoolExecutor
        """
        if not self.model_loaded:
            raise Exception("Model not properly loaded")
        
        results = {}
        total_chunks = len(chunks)
        
        print(f"🚀 Processing {total_chunks} chunks with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self.transcribe_single_chunk, chunk, sample_rate): chunk[0] 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    chunk_idx, transcription = future.result()
                    results[chunk_idx] = transcription
                except Exception as e:
                    print(f"Error in future: {e}")
        
        # Reconstruct transcript in correct order
        full_transcript = " ".join(results[i] for i in sorted(results.keys()) if results[i])
        
        return full_transcript.strip() if full_transcript else "No clear speech detected in the audio file."

    def transcribe_chunks_sequential(self, chunks, sample_rate=16000):
        """
        Transcribe audio chunks sequentially (fallback method)
        """
        if not self.model_loaded:
            raise Exception("Model not properly loaded")
            
        full_transcript = ""
        
        for chunk_data in chunks:
            chunk_idx, transcription = self.transcribe_single_chunk(chunk_data, sample_rate)
            if transcription:
                full_transcript += transcription + " "
        
        result = full_transcript.strip()
        return result if result else "No clear speech detected in the audio file."

    def transcribe(self, file_path, chunk_size_s=30, use_parallel=True):
        """
        Main transcription method with optional parallel processing
        
        Args:
            file_path: Path to audio file
            chunk_size_s: Duration of each chunk in seconds
            use_parallel: Whether to use parallel processing (default: True)
        """
        try:
            if self.model_loaded:
                chunks = self.load_audio_chunks(file_path, chunk_duration=chunk_size_s, overlap=1.5)
                
                if not chunks:
                    return "No audio content detected in the file."
                
                if use_parallel and self.max_workers > 1:
                    return self.transcribe_chunks_parallel(chunks)
                else:
                    return self.transcribe_chunks_sequential(chunks)
            else:
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
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            max_amplitude = np.max(np.abs(audio))
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'channels': 1,
                'max_amplitude': float(max_amplitude),
                'has_audio': max_amplitude > 0.001
            }
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")

    def preprocess_audio(self, input_path, output_path=None):
        """
        Preprocess audio with noise reduction and filtering
        """
        try:
            audio, sr = librosa.load(input_path, sr=16000)

            # Noise reduction
            audio = nr.reduce_noise(y=audio, sr=sr)

            # High-pass filter
            b, a = sps.butter(3, 100/(sr/2), btype='highpass')
            audio = sps.lfilter(b, a, audio)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Save
            if output_path:
                sf.write(output_path, audio, sr)
                return output_path

            temp_dir = mkdtemp()
            temp_path = os.path.join(temp_dir, "processed_audio.wav")
            sf.write(temp_path, audio, sr)
            return temp_path

        except Exception as e:
            raise Exception(f"Audio preprocessing error: {str(e)}")