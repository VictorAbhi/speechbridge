import os
import json
from pydub import AudioSegment
import argparse

def chunk_audio_flat_structure(file_path, chunk_length_ms=30000, output_format="wav", output_dir="whisper_chunks"):
    """
    Split audio file into 30-second chunks in flat structure
    - If audio <= 30s: just copy with original name
    - If audio > 30s: chunk and name as audio_1_chunk*.wav
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Get file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate total duration
        duration_ms = len(audio)
        
        chunks = []
        
        # If audio is shorter or equal to 30 seconds, just copy it
        if duration_ms <= chunk_length_ms:
            output_name = f"{file_name}.{output_format}"
            output_path = os.path.join(output_dir, output_name)
            
            print(f"Audio is {duration_ms/1000:.1f}s (<= 30s), copying as: {output_name}")
            audio.export(output_path, format=output_format)
            
            chunks.append({
                "file_name": output_name,
                "original_duration": duration_ms / 1000,
                "padded": False,
                "chunked": False
            })
            
        else:
            # Audio is longer than 30 seconds, chunk it
            start_ms = 0
            chunk_index = 1
            
            while start_ms < duration_ms:
                end_ms = min(start_ms + chunk_length_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Pad chunks shorter than 30 seconds with silence
                if len(chunk) < chunk_length_ms:
                    silence_duration = chunk_length_ms - len(chunk)
                    silence = AudioSegment.silent(duration=silence_duration)
                    chunk = chunk + silence
                
                # Name format: audio_1_chunk1.wav, audio_1_chunk2.wav, etc.
                chunk_name = f"{file_name}_chunk{chunk_index}.{output_format}"
                chunk_path = os.path.join(output_dir, chunk_name)
                
                print(f"Exporting {chunk_name} (original: {len(chunk)/1000:.1f}s)")
                chunk.export(chunk_path, format=output_format)
                
                chunks.append({
                    "file_name": chunk_name,
                    "original_duration": (end_ms - start_ms) / 1000,
                    "padded": len(chunk) > (end_ms - start_ms),
                    "chunked": True
                })
                
                chunk_index += 1
                start_ms += chunk_length_ms
            
            print(f"Created {len(chunks)} chunks for {file_name}")
        
        return chunks
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def create_whisper_metadata_template(chunks_data, output_file="metadata.json"):
    """
    Create Whisper metadata template WITHOUT transcriptions
    """
    training_data = []
    
    for chunk_info in chunks_data:
        training_data.append({
            "file_name": chunk_info["file_name"],
            "text": "TODO: ADD TRANSCRIPTION HERE",
            "original_duration": chunk_info["original_duration"],
            "padded": chunk_info["padded"],
            "chunked": chunk_info["chunked"],
            "needs_transcription": True
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata template saved to {output_file}")

def convert_mp3_to_wav_if_needed(directory="."):
    """
    Optional: Convert existing MP3 files to WAV for better quality
    """
    import glob
    
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    for mp3_file in mp3_files:
        wav_file = mp3_file.replace('.mp3', '.wav')
        if not os.path.exists(wav_file):
            print(f"Converting {mp3_file} to WAV...")
            audio = AudioSegment.from_mp3(mp3_file)
            audio.export(wav_file, format="wav")

def process_audio_files_flat(directory=".", pattern="audio_*.wav", output_dir="whisper_chunks"):
    """
    Process all audio files and create flat structure for Colab
    """
    import glob
    
    # Find all files matching the pattern
    search_pattern = os.path.join(directory, pattern)
    audio_files = glob.glob(search_pattern)
    
    if not audio_files:
        print(f"No WAV files found matching {pattern}, checking for MP3...")
        search_pattern = os.path.join(directory, "audio_*.mp3")
        audio_files = glob.glob(search_pattern)
        
        if audio_files:
            print("Found MP3 files. Consider converting to WAV for better quality.")
            response = input("Convert MP3 to WAV first? (y/n): ")
            if response.lower() == 'y':
                convert_mp3_to_wav_if_needed(directory)
                audio_files = glob.glob(os.path.join(directory, "audio_*.wav"))
    
    if not audio_files:
        print(f"No audio files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    
    all_chunks_data = []
    
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file}")
        chunks_data = chunk_audio_flat_structure(audio_file, output_dir=output_dir)
        all_chunks_data.extend(chunks_data)
    
    # Create metadata template
    if all_chunks_data:
        create_whisper_metadata_template(all_chunks_data)
        
        # Print summary
        total_chunks = len(all_chunks_data)
        short_files = sum(1 for chunk in all_chunks_data if not chunk["chunked"])
        chunked_files = sum(1 for chunk in all_chunks_data if chunk["chunked"])
        
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total WAV files created: {total_chunks}")
        print(f"Short files (<=30s): {short_files}")
        print(f"Chunked files (>30s): {chunked_files}")
        print(f"All files saved in: {output_dir}/")
        print(f"Format: WAV (recommended for Whisper fine-tuning)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chunk audio files for Whisper fine-tuning')
    parser.add_argument('--input_dir', default='.', help='Input directory with audio files')
    parser.add_argument('--output_dir', default='whisper_chunks', help='Output directory for chunks')
    parser.add_argument('--pattern', default='audio_*.wav', help='File pattern to match')
    
    args = parser.parse_args()
    
    process_audio_files_flat(
        directory=args.input_dir,
        pattern=args.pattern,
        output_dir=args.output_dir
    )