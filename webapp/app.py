from flask import Flask, jsonify, request, render_template, session, redirect, url_for, Response
from auth import AuthHandler
from summarizer import SummarizerService
from transcription import TranscriptionService
from functools import wraps
import os
from werkzeug.utils import secure_filename
import json
import time
import uuid
import threading, re
import subprocess

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-change-this"  # change to a secure random key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CONVERTED_FOLDER'] = 'converted'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERTED_FOLDER'], exist_ok=True)

# Initialize services
auth_handler = AuthHandler()
# Transcription services will be created on-demand based on language
transcription_services = {}  # Cache for language-specific services

# Global dictionary to store progress for each session
progress_data = {}

def get_transcription_service(language="en"):
    """Get or create a transcription service for the specified language"""
    if language not in transcription_services:
        transcription_services[language] = TranscriptionService(language=language)
    return transcription_services[language]

# ----------------------
# FFmpeg Conversion Functions
# ----------------------
def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def parse_ffmpeg_progress(line):
    """Parse FFmpeg stderr output to extract progress information"""
    # FFmpeg outputs progress like: "time=00:01:23.45"
    time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        seconds = float(time_match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    return None

def get_audio_duration(input_path):
    """Get duration of audio/video file using ffprobe"""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Could not get duration: {e}")
    return None

def check_audio_format(input_path):
    """
    Check if audio file already meets our requirements
    Returns True if file is already in correct format (WAV, 16kHz, mono, PCM)
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name,sample_rate,channels',
            '-of', 'json',
            input_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                codec = stream.get('codec_name', '')
                sample_rate = int(stream.get('sample_rate', 0))
                channels = int(stream.get('channels', 0))
                
                # Check if it's already WAV with correct specs
                # codec_name for WAV is usually 'pcm_s16le' or similar PCM variant
                is_pcm = codec.startswith('pcm_')
                is_16khz = sample_rate == 16000
                is_mono = channels == 1
                
                if is_pcm and is_16khz and is_mono:
                    return True
                    
    except Exception as e:
        print(f"Error checking audio format: {e}")
    
    return False

def convert_to_wav(input_path, output_path, session_id=None):
    """
    Convert audio/video file to WAV format using FFmpeg with progress updates
    Skips conversion if file is already in correct format
    """
    try:
        if session_id:
            update_progress(session_id, 'convert', 25, 'Checking audio format...', 
                          'Analyzing file specifications')
        
        # Check if file is already in correct format
        if check_audio_format(input_path):
            if session_id:
                update_progress(session_id, 'convert', 40, 'Format check complete!', 
                              'File is already in correct format - skipping conversion')
            # Copy file instead of converting
            import shutil
            shutil.copy2(input_path, output_path)
            return True
        
        if session_id:
            update_progress(session_id, 'convert', 27, 'Starting conversion...', 
                          'Converting to required format')
        
        # Get file duration for progress calculation
        duration = get_audio_duration(input_path)
        
        # FFmpeg command
        command = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        # Start FFmpeg process with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Monitor stderr for progress (FFmpeg writes progress to stderr)
        last_update_time = time.time()
        
        while True:
            line = process.stderr.readline()
            if not line:
                break
            
            # Parse progress from FFmpeg output
            if session_id and duration:
                current_time = parse_ffmpeg_progress(line)
                if current_time:
                    progress_percent = min((current_time / duration) * 100, 99)
                    # Map to 25-40% range for conversion stage
                    mapped_progress = 25 + (progress_percent * 0.15)
                    
                    # Update every 2 seconds to avoid spam
                    if time.time() - last_update_time > 2:
                        update_progress(
                            session_id, 
                            'convert', 
                            int(mapped_progress), 
                            f'Converting to WAV format... ({progress_percent:.0f}%)',
                            f'Processing: {int(current_time)}s / {int(duration)}s'
                        )
                        last_update_time = time.time()
            elif session_id:
                # If we don't have duration, send keepalive updates
                if time.time() - last_update_time > 3:
                    update_progress(
                        session_id, 
                        'convert', 
                        30, 
                        'Converting to WAV format...',
                        'Processing audio data...'
                    )
                    last_update_time = time.time()
        
        # Wait for process to complete
        return_code = process.wait(timeout=300)
        
        if return_code == 0:
            if session_id:
                update_progress(session_id, 'convert', 40, 'Conversion complete!', 
                              'File ready for transcription')
            return True
        else:
            stderr = process.stderr.read()
            print(f"FFmpeg conversion error: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FFmpeg conversion timed out")
        if session_id:
            update_progress(session_id, 'error', 0, 'Error', 
                          'Conversion timed out - file may be too large')
        return False
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        if session_id:
            update_progress(session_id, 'error', 0, 'Error', 
                          f'Conversion failed: {str(e)}')
        return False

def extract_audio_from_video(input_path, output_path, session_id=None):
    """
    Extract audio from video file with progress updates
    """
    try:
        if session_id:
            update_progress(session_id, 'extract', 20, 'Extracting audio from video...', 
                          'Starting audio extraction')
        
        # Get video duration
        duration = get_audio_duration(input_path)
        
        command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        # Start FFmpeg process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress
        last_update_time = time.time()
        
        while True:
            line = process.stderr.readline()
            if not line:
                break
            
            if session_id and duration:
                current_time = parse_ffmpeg_progress(line)
                if current_time:
                    progress_percent = min((current_time / duration) * 100, 99)
                    # Map to 20-35% range for extraction stage
                    mapped_progress = 20 + (progress_percent * 0.15)
                    
                    if time.time() - last_update_time > 2:
                        update_progress(
                            session_id, 
                            'extract', 
                            int(mapped_progress), 
                            f'Extracting audio... ({progress_percent:.0f}%)',
                            f'Extracting: {int(current_time)}s / {int(duration)}s'
                        )
                        last_update_time = time.time()
            elif session_id:
                if time.time() - last_update_time > 3:
                    update_progress(
                        session_id, 
                        'extract', 
                        25, 
                        'Extracting audio from video...',
                        'Processing video file...'
                    )
                    last_update_time = time.time()
        
        return_code = process.wait(timeout=300)
        
        if return_code == 0:
            if session_id:
                update_progress(session_id, 'extract', 35, 'Extraction complete!', 
                              'Audio extracted successfully')
            return True
        else:
            stderr = process.stderr.read()
            print(f"FFmpeg audio extraction error: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        if session_id:
            update_progress(session_id, 'error', 0, 'Error', 
                          f'Extraction failed: {str(e)}')
        return False

# ----------------------
# Helper Functions
# ----------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def update_progress(session_id, stage, progress, message, details="", partial_text=""):
    """Update progress for a specific session"""
    progress_data[session_id] = {
        'stage': stage,
        'progress': progress,
        'message': message,
        'details': details,
        'partial_text': partial_text,
        'timestamp': time.time()
    }

def transcribe_with_progress(transcribe_func, filepath, session_id, start_progress, end_progress, transcription_service=None):
    """Wrapper function to provide progress updates during transcription"""
    
    # If it's the chunked transcription method, we can provide real progress
    if transcription_service and hasattr(transcription_service, 'transcribe_with_callback'):
        # Custom transcription with callback for real chunks
        def progress_callback(chunk_text, progress_percent):
            current_progress = start_progress + (progress_percent * (end_progress - start_progress) / 100)
            partial_transcription = progress_data[session_id].get('partial_transcription', '') + chunk_text + ' '
            progress_data[session_id]['partial_transcription'] = partial_transcription
            update_progress(session_id, 'transcribe', current_progress, 
                          f'Transcribing speech... ({progress_percent:.0f}%)',
                          f'Processing audio chunk {int(progress_percent/10)+1}', 
                          partial_transcription)
        
        return transcription_service.transcribe_with_callback(filepath, progress_callback)
    else:
        # Simulate chunked progress for existing methods
        result = simulate_chunked_transcription(transcribe_func, filepath, session_id, start_progress, end_progress)
        return result

def simulate_chunked_transcription(transcribe_func, filepath, session_id, start_progress, end_progress):
    """Simulate chunked transcription with progress updates"""
    
    # Start transcription in background thread
    transcription_result = {'text': ''}
    transcription_error = {'error': None}
    
    def run_transcription():
        try:
            transcription_result['text'] = transcribe_func(filepath)
        except Exception as e:
            transcription_error['error'] = e
    
    # Start transcription thread
    transcription_thread = threading.Thread(target=run_transcription)
    transcription_thread.start()
    
    # Simulate progress with sample text chunks
    sample_chunks = [
        "Processing audio input...",
        "Detecting speech patterns...", 
        "Converting audio to text...",
        "Analyzing speech segments...",
        "Generating transcription...",
        "Finalizing results...",
    ]
    
    chunk_progress = start_progress
    progress_increment = (end_progress - start_progress) / len(sample_chunks)
    
    for i, chunk in enumerate(sample_chunks):
        if transcription_error['error']:
            raise transcription_error['error']
            
        chunk_progress += progress_increment
        partial_text = progress_data[session_id].get('partial_transcription', '') + chunk + ' '
        progress_data[session_id]['partial_transcription'] = partial_text
        
        update_progress(session_id, 'transcribe', chunk_progress, 
                       f'Transcribing speech... ({((i+1)/len(sample_chunks)*100):.0f}%)',
                       f'Processing segment {i+1} of {len(sample_chunks)}', 
                       partial_text)
        
        time.sleep(0.8)  # Simulate processing time
        
        # Check if transcription is complete
        if not transcription_thread.is_alive():
            break
    
    # Wait for transcription to complete
    transcription_thread.join()
    
    if transcription_error['error']:
        raise transcription_error['error']
    
    return transcription_result['text']

def process_transcription_async(filepath, filename, session_id, language="en"):
    """Process transcription in background thread with progress updates"""
    converted_path = None
    
    try:
        # Get language-specific transcription service
        transcription_service = get_transcription_service(language)
        
        # Stage 1: Analyze audio
        update_progress(session_id, 'analyze', 15, 'Analyzing file...', 'Checking file format and type')
        
        # Check file extension to determine if conversion is needed
        file_ext = os.path.splitext(filepath)[1].lower()
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Determine processing path
        processing_path = filepath
        
        # Only convert if not already in correct format
        if file_ext == '.wav':
            # Check if WAV is already in correct format (16kHz, mono, PCM)
            update_progress(session_id, 'analyze', 20, 'Checking WAV format...', 
                          'Verifying audio specifications')
            
            if check_audio_format(filepath):
                update_progress(session_id, 'analyze', 40, 'Format verified!', 
                              'WAV file is ready for transcription')
                processing_path = filepath
            else:
                # WAV exists but wrong specs - need to convert
                update_progress(session_id, 'convert', 25, 'Converting WAV format...', 
                              'Adjusting sample rate and channels')
                converted_filename = f"{uuid.uuid4()}.wav"
                converted_path = os.path.join(app.config['CONVERTED_FOLDER'], converted_filename)
                
                success = convert_to_wav(filepath, converted_path, session_id)
                if not success:
                    update_progress(session_id, 'error', 0, 'Error', 
                                  'Failed to convert WAV file to required format.')
                    return
                processing_path = converted_path
        
        elif file_ext in audio_extensions or file_ext in video_extensions:
            # Non-WAV format - needs conversion
            update_progress(session_id, 'convert', 25, 'Converting audio format...', 
                          f'Converting {file_ext} to WAV format')
            
            # Generate output path for converted file
            converted_filename = f"{uuid.uuid4()}.wav"
            converted_path = os.path.join(app.config['CONVERTED_FOLDER'], converted_filename)
            
            # Check if it's a video file
            if file_ext in video_extensions:
                success = extract_audio_from_video(filepath, converted_path, session_id)
            else:
                success = convert_to_wav(filepath, converted_path, session_id)
            
            if not success:
                update_progress(session_id, 'error', 0, 'Error', 
                              f'Failed to convert {file_ext} file. Please ensure FFmpeg is installed.')
                return
            
            processing_path = converted_path
        else:
            # Unknown format
            update_progress(session_id, 'error', 0, 'Error', 
                          f'Unsupported file format: {file_ext}')
            return
        
        # Stage 2: Analyze audio properties
        update_progress(session_id, 'analyze', 45, 'Analyzing audio properties...', 
                       'Checking audio quality and duration')
        
        audio_info = transcription_service.get_audio_info(processing_path)
        duration_str = f"{audio_info['duration']:.1f}s"
        file_size = os.path.getsize(processing_path)
        file_size_str = f"{file_size/1024/1024:.1f}MB"
        
        # Check if audio has meaningful content
        if not audio_info['has_audio']:
            update_progress(session_id, 'error', 0, 'Error', 'No audio content detected in file')
            return
        
        # Stage 3: Processing
        update_progress(session_id, 'process', 55, 'Preparing for transcription...', 
                       'Optimizing audio for speech recognition')
        time.sleep(0.5)
        
        # Determine processing method based on file size and duration
        max_simple_size = 5 * 1024 * 1024  # 5MB threshold
        max_simple_duration = 60  # 60 seconds threshold
        
        # Stage 4: Transcription with real-time chunks
        update_progress(session_id, 'transcribe', 70, 'Transcribing speech...', 
                       'AI is converting speech to text')
        
        # Initialize partial transcription
        progress_data[session_id]['partial_transcription'] = ""
        
        try:
            if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                # For small files, use simple method
                transcription = transcribe_with_progress(
                    transcription_service.transcribe_simple, 
                    processing_path, session_id, 70, 85, transcription_service
                )
                method = "Direct processing (small file)"
            else:
                # Use chunked transcription for larger files
                transcription = transcribe_with_progress(
                    transcription_service.transcribe, 
                    processing_path, session_id, 70, 85, transcription_service
                )
                method = "Chunked processing"
        except Exception as transcribe_error:
            # Fallback method
            update_progress(session_id, 'transcribe', 75, 'Trying alternative method...', 
                          'Primary method failed, using fallback')
            try:
                if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                    transcription = transcribe_with_progress(
                        transcription_service.transcribe, 
                        processing_path, session_id, 75, 85, transcription_service
                    )
                    method = "Chunked processing (fallback)"
                else:
                    transcription = transcribe_with_progress(
                        transcription_service.transcribe_simple, 
                        processing_path, session_id, 75, 85, transcription_service
                    )
                    method = "Direct processing (fallback)"
            except Exception as fallback_error:
                try:
                    update_progress(session_id, 'transcribe', 78, 'Preprocessing audio...', 
                                  'Applying audio enhancements')
                    processed_path = transcription_service.preprocess_audio(processing_path)
                    transcription = transcribe_with_progress(
                        transcription_service.transcribe, 
                        processed_path, session_id, 78, 85, transcription_service
                    )
                    method = "Preprocessed + chunked"
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    raise transcribe_error
        
        # Stage 5: Summarization
        update_progress(session_id, 'summarize', 90, 'Generating summary...', 
                       'Creating intelligent summary of content')
        
        summarizer = SummarizerService()
        summary = summarizer.summarize(transcription, language=language) if transcription else ""
        
        # Check if transcription is meaningful
        if not transcription or transcription.lower() in ["no speech detected.", 
                                                          "no clear speech detected in the audio file.", ""]:
            update_progress(session_id, 'error', 0, 'Error', 
                          'No speech could be detected in the audio file')
            return
        
        # Stage 6: Complete
        result = {
            'transcription': transcription,
            'summary': summary,
            'filename': filename,
            'duration': duration_str,
            'method': method,
            'file_size': file_size_str,
            'audio_quality': "Good" if audio_info['max_amplitude'] > 0.1 else "Low volume detected",
            'converted': converted_path is not None,
            'format_check': 'Skipped conversion - correct format' if converted_path is None and file_ext == '.wav' else 'Converted'
        }
        
        update_progress(session_id, 'complete', 100, 'Complete!', 'Transcription finished successfully')
        progress_data[session_id]['result'] = result
        
    except Exception as e:
        error_msg = str(e)
        # Provide more user-friendly error messages
        if "No such file" in error_msg or "cannot identify image file" in error_msg:
            error_msg = "File upload failed or file is corrupted. Please try again with a different file."
        elif "not supported" in error_msg.lower() or "format" in error_msg.lower():
            error_msg = "Audio format not supported or file is corrupted. Please try a different file."
        elif "memory" in error_msg.lower():
            error_msg = "File too large or system out of memory. Please try a smaller file or shorter audio clip."
        elif "Model not properly loaded" in error_msg:
            error_msg = "Transcription service not available. Please try again later."
        elif "No audio content" in error_msg:
            error_msg = "The uploaded file doesn't contain detectable audio. Please check your file and try again."
        elif "ffmpeg" in error_msg.lower():
            error_msg = "FFmpeg is not installed or not accessible. Please install FFmpeg to process this file type."
        
        update_progress(session_id, 'error', 0, 'Error', f"Transcription failed: {error_msg}")
        print(f"Transcription error for session {session_id}: {error_msg}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to remove {filepath}: {e}")
        
        # Clean up converted file (only if we created one)
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception as e:
                print(f"Failed to remove {converted_path}: {e}")
# ----------------------
# Routes
# ----------------------

# Dashboard page (language selection)
@app.route("/", methods=["GET"])
@login_required
def dashboard():
    return render_template("dashboard.html")

# Old index route - kept for backward compatibility
@app.route("/index", methods=["GET"])
@login_required
def index():
    return render_template("index.html")

# English transcription page
@app.route("/transcribe/english", methods=["GET"])
@login_required
def transcribe_english_page():
    return render_template("transcribe_english.html")

# Nepali transcription page
@app.route("/transcribe/nepali", methods=["GET"])
@login_required
def transcribe_nepali_page():
    return render_template("transcribe_nepali.html")

# GET login page
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

# POST login credentials
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        return render_template("login.html", error="Missing username or password")
    
    if auth_handler.login(username, password):
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        return render_template("login.html", error="Invalid credentials")

# GET register page
@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

# POST register credentials
@app.route("/register", methods=["POST"])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        return render_template("register.html", error="Missing username or password")
    
    if auth_handler.register(username, password):
        return render_template("login.html", success="Registration successful! Please login.")
    else:
        return render_template("register.html", error="Username already exists")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# Check FFmpeg availability
@app.route("/check_ffmpeg", methods=["GET"])
@login_required
def check_ffmpeg_endpoint():
    available = check_ffmpeg()
    return jsonify({"available": available})

# Start transcription with file upload
@app.route("/transcribe", methods=["POST"])
@login_required
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check file extension - now supporting video files too
    allowed_extensions = {
        # Audio formats
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm',
        # Video formats (audio will be extracted)
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'
    }
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({
            "error": f"Unsupported file format. Supported formats: {', '.join(sorted(allowed_extensions))}"
        }), 400
    
    # Check if FFmpeg is available for non-WAV files
    if file_ext != '.wav' and not check_ffmpeg():
        return jsonify({
            "error": "FFmpeg is required to process this file format but is not installed. Please install FFmpeg or upload a WAV file."
        }), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Get language parameter (default to English)
    language = request.form.get('language', 'en')
    if language not in ['en', 'ne']:
        return jsonify({"error": "Invalid language. Use 'en' or 'ne'"}), 400
    
    try:
        # Save uploaded file
        file.save(filepath)
        
        # Generate unique session ID for this transcription
        session_id = str(uuid.uuid4())
        
        # Initialize progress
        lang_name = "English" if language == "en" else "Nepali"
        update_progress(session_id, 'upload', 10, 'File uploaded successfully', 
                       f'Starting {lang_name} transcription process')
        
        # Start background processing with language parameter
        thread = threading.Thread(
            target=process_transcription_async,
            args=(filepath, filename, session_id, language)
        )
        thread.start()
        
        return jsonify({"session_id": session_id}), 200
        
    except Exception as e:
        # Clean up uploaded file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

# Server-Sent Events endpoint for progress updates
@app.route('/progress/<session_id>')
@login_required
def progress_stream(session_id):
    def event_stream():
        while True:
            if session_id in progress_data:
                data = progress_data[session_id]
                yield f"data: {json.dumps(data)}\n\n"
                
                # If complete or error, clean up and close connection
                if data['stage'] in ['complete', 'error']:
                    # Keep data for a short while then clean up
                    time.sleep(2)
                    if session_id in progress_data:
                        del progress_data[session_id]
                    break
            else:
                # Send heartbeat if no data yet
                yield f"data: {json.dumps({'stage': 'waiting', 'progress': 0, 'message': 'Initializing...'})}\n\n"
            
            time.sleep(0.5)  # Update every 500ms
    
    return Response(event_stream(), mimetype='text/event-stream')

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # Check FFmpeg on startup
    if not check_ffmpeg():
        print("\nWARNING: FFmpeg is not installed or not in PATH!")
        print("Only WAV files will be supported without FFmpeg.")
        print("To enable all formats, install FFmpeg:")
        print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - MacOS: brew install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html\n")
    else:
        print("FFmpeg is available - all audio/video formats supported!\n")
    
    app.run(debug=True, threaded=True)  # Enable threading for SSE