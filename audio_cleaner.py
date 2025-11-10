import os
import json
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import noisereduce as nr
import argparse
import librosa
import soundfile as sf
from scipy import signal

class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr  # Whisper uses 16kHz
        
    def load_audio(self, file_path):
        """Load audio file with proper sampling rate"""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def detect_silence(self, audio, threshold_db=-40, min_silence_len=500):
        """Detect silent segments in audio"""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find silent regions
        silent_regions = audio_db < threshold_db
        
        # Group consecutive silent samples
        silent_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_regions):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = i
            elif not is_silent and in_silence:
                in_silence = False
                silence_duration = (i - silence_start) / self.target_sr * 1000  # ms
                if silence_duration >= min_silence_len:
                    silent_segments.append((silence_start, i))
        
        return silent_segments
    
    def remove_silence(self, audio, threshold_db=-35, min_silence_len=300, padding=100):
        """Remove prolonged silence from audio"""
        silent_segments = self.detect_silence(audio, threshold_db, min_silence_len)
        
        if not silent_segments:
            return audio
        
        # Create mask for non-silent regions
        mask = np.ones(len(audio), dtype=bool)
        
        for start, end in silent_segments:
            # Add padding to avoid cutting speech
            start_padded = max(0, start - int(padding * self.target_sr / 1000))
            end_padded = min(len(audio), end + int(padding * self.target_sr / 1000))
            mask[start_padded:end_padded] = False
        
        return audio[mask]
    
    def reduce_noise(self, audio, stationary=True):
        """Reduce background noise using noisereduce"""
        try:
            if stationary:
                # For stationary noise
                reduced_audio = nr.reduce_noise(y=audio, sr=self.target_sr, stationary=stationary)
            else:
                # For non-stationary noise, use first 500ms as noise sample
                noise_sample = audio[:int(0.5 * self.target_sr)]
                reduced_audio = nr.reduce_noise(y=audio, sr=self.target_sr, y_noise=noise_sample)
            
            return reduced_audio
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def apply_bandpass_filter(self, audio, low_cutoff=80, high_cutoff=8000):
        """Apply bandpass filter to remove extreme frequencies - FIXED VERSION"""
        nyquist = self.target_sr / 2
        
        # Normalize frequencies to 0-1 range (required by scipy)
        low_normalized = low_cutoff / nyquist
        high_normalized = high_cutoff / nyquist
        
        # Ensure frequencies are within valid range
        low_normalized = max(0.001, min(0.499, low_normalized))
        high_normalized = max(0.002, min(0.499, high_normalized))
        
        if low_normalized >= high_normalized:
            # Fallback to sensible values if invalid
            low_normalized = 80 / nyquist
            high_normalized = 8000 / nyquist
        
        b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def normalize_audio(self, audio, target_dBFS=-20):
        """Normalize audio to target loudness"""
        # Convert to pydub AudioSegment for normalization
        audio_int = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=self.target_sr,
            sample_width=audio_int.dtype.itemsize,
            channels=1
        )
        
        # Normalize
        normalized = normalize(audio_segment, headroom=0.1)
        
        # Convert back to numpy
        normalized_audio = np.array(normalized.get_array_of_samples()).astype(np.float32) / 32767.0
        
        return normalized_audio
    
    def compress_audio(self, audio, threshold=-20.0, ratio=2.0):
        """Apply dynamic range compression"""
        audio_int = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=self.target_sr,
            sample_width=audio_int.dtype.itemsize,
            channels=1
        )
        
        compressed = compress_dynamic_range(audio_segment, threshold=threshold, ratio=ratio)
        compressed_audio = np.array(compressed.get_array_of_samples()).astype(np.float32) / 32767.0
        
        return compressed_audio
    
    def detect_voice_activity(self, audio, threshold=0.025):
        """Simple voice activity detection"""
        energy = librosa.feature.rms(y=audio)[0]
        voice_segments = energy > threshold
        
        # Find speech regions
        speech_regions = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(voice_segments):
            if is_speech and not in_speech:
                in_speech = True
                speech_start = i * len(audio) // len(energy)
            elif not is_speech and in_speech:
                in_speech = False
                speech_end = i * len(audio) // len(energy)
                speech_regions.append((speech_start, speech_end))
        
        return speech_regions
    
    def preprocess_audio(self, file_path, aggressive_cleaning=False):
        """Complete audio preprocessing pipeline"""
        print(f"Preprocessing: {os.path.basename(file_path)}")
        
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        original_length = len(audio)
        
        # 1. Remove silence (aggressive for noisy audio)
        silence_threshold = -40 if aggressive_cleaning else -35
        audio = self.remove_silence(audio, threshold_db=silence_threshold)
        print(f"  - Silence removal: {original_length} -> {len(audio)} samples")
        
        # 2. Noise reduction
        audio = self.reduce_noise(audio, stationary=not aggressive_cleaning)
        print(f"  - Noise reduction applied")
        
        # 3. Bandpass filter (remove extreme frequencies) - NOW FIXED
        audio = self.apply_bandpass_filter(audio)
        print(f"  - Bandpass filtering applied")
        
        # 4. Normalize audio
        audio = self.normalize_audio(audio)
        print(f"  - Audio normalized")
        
        # 5. Apply compression if aggressive cleaning
        if aggressive_cleaning:
            audio = self.compress_audio(audio)
            print(f"  - Dynamic compression applied")
        
        # Check if audio is too short after cleaning
        if len(audio) < 0.5 * self.target_sr:  # Less than 0.5 seconds
            print(f"  ⚠️  Warning: Audio too short after cleaning ({len(audio)/self.target_sr:.1f}s)")
            return None
        
        return audio

def chunk_audio_with_preprocessing(file_path, preprocessor, output_dir="cleaned_chunks", 
                                 chunk_length=30, aggressive_cleaning=False):
    """Chunk audio with extensive preprocessing"""
    
    # Preprocess entire audio first
    cleaned_audio = preprocessor.preprocess_audio(file_path, aggressive_cleaning)
    
    if cleaned_audio is None:
        print(f"Failed to preprocess {file_path}")
        return []
    
    # Get file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    chunks = []
    sr = preprocessor.target_sr
    chunk_samples = chunk_length * sr
    
    # Calculate total duration after cleaning
    duration_seconds = len(cleaned_audio) / sr
    
    # If audio is shorter or equal to chunk length, save as single file
    if duration_seconds <= chunk_length:
        output_name = f"{file_name}.wav"
        output_path = os.path.join(output_dir, output_name)
        
        sf.write(output_path, cleaned_audio, sr)
        
        chunks.append({
            "file_name": output_name,
            "original_duration": duration_seconds,
            "padded": False,
            "chunked": False,
            "cleaned": True
        })
        
        print(f"  - Saved as single file: {output_name}")
        
    else:
        # Split into chunks
        num_chunks = int(np.ceil(len(cleaned_audio) / chunk_samples))
        
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(cleaned_audio))
            
            chunk_audio = cleaned_audio[start_sample:end_sample]
            
            # Pad if necessary
            if len(chunk_audio) < chunk_samples:
                padding = chunk_samples - len(chunk_audio)
                chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant')
            
            chunk_name = f"{file_name}_chunk{i+1}.wav"
            chunk_path = os.path.join(output_dir, chunk_name)
            
            sf.write(chunk_path, chunk_audio, sr)
            
            chunks.append({
                "file_name": chunk_name,
                "original_duration": (end_sample - start_sample) / sr,
                "padded": len(chunk_audio) > (end_sample - start_sample),
                "chunked": True,
                "cleaned": True
            })
        
        print(f"  - Created {num_chunks} cleaned chunks")
    
    return chunks

def analyze_audio_quality(file_path):
    """Analyze audio quality and suggest cleaning level"""
    preprocessor = AudioPreprocessor()
    audio, sr = preprocessor.load_audio(file_path)
    
    if audio is None:
        return "poor", 0
    
    # Calculate metrics
    rms_energy = np.mean(librosa.feature.rms(y=audio)[0])
    silence_segments = preprocessor.detect_silence(audio, threshold_db=-35)
    silence_ratio = sum([end-start for start,end in silence_segments]) / len(audio) if silence_segments else 0
    
    print(f"Quality analysis for {os.path.basename(file_path)}:")
    print(f"  - RMS Energy: {rms_energy:.4f}")
    print(f"  - Silence ratio: {silence_ratio:.2%}")
    
    # Determine cleaning aggressiveness
    if silence_ratio > 0.3 or rms_energy < 0.01:
        return "aggressive", silence_ratio
    elif silence_ratio > 0.15 or rms_energy < 0.02:
        return "moderate", silence_ratio
    else:
        return "light", silence_ratio

def main():
    parser = argparse.ArgumentParser(description='Advanced audio preprocessing for Whisper fine-tuning')
    parser.add_argument('--input_dir', default='.', help='Input directory with audio files')
    parser.add_argument('--output_dir', default='cleaned_chunks', help='Output directory for cleaned chunks')
    parser.add_argument('--pattern', default='audio_*.wav', help='File pattern to match')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive cleaning for all files')
    parser.add_argument('--auto_clean', action='store_true', help='Automatically determine cleaning level')
    
    args = parser.parse_args()
    
    import glob
    
    # Find audio files
    search_pattern = os.path.join(args.input_dir, args.pattern)
    audio_files = glob.glob(search_pattern)
    
    if not audio_files:
        print(f"No files found matching: {search_pattern}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print("Starting advanced preprocessing...")
    
    preprocessor = AudioPreprocessor()
    all_chunks = []
    
    for file_path in audio_files:
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Determine cleaning level
        if args.aggressive:
            cleaning_level = "aggressive"
        elif args.auto_clean:
            cleaning_level, silence_ratio = analyze_audio_quality(file_path)
            print(f"  - Recommended cleaning: {cleaning_level}")
        else:
            cleaning_level = "moderate"
        
        aggressive_cleaning = (cleaning_level == "aggressive")
        
        # Process audio
        chunks = chunk_audio_with_preprocessing(
            file_path, preprocessor, args.output_dir,
            aggressive_cleaning=aggressive_cleaning
        )
        all_chunks.extend(chunks)
    
    # Create metadata
    if all_chunks:
        metadata = []
        for chunk in all_chunks:
            metadata.append({
                "file_name": chunk["file_name"],
                "text": "TODO: ADD TRANSCRIPTION HERE",
                "original_duration": chunk["original_duration"],
                "cleaned": chunk["cleaned"]
            })
        
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print("PREPROCESSING COMPLETE!")
        print(f"Total cleaned chunks: {len(all_chunks)}")
        print(f"Output directory: {args.output_dir}")
        print(f"Metadata file: {metadata_path}")
        print(f"All audio cleaned and ready for Whisper fine-tuning! ✅")

if __name__ == "__main__":
    main()