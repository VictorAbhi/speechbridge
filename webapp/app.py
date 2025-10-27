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
import threading

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-change-this"  # change to a secure random key
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
auth_handler = AuthHandler()
transcription_service = TranscriptionService()
summarizer = SummarizerService()

# Global dictionary to store progress for each session
progress_data = {}

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

def transcribe_with_progress(transcribe_func, filepath, session_id, start_progress, end_progress):
    """Wrapper function to provide progress updates during transcription"""
    
    # If it's the chunked transcription method, we can provide real progress
    if hasattr(transcription_service, 'transcribe_with_callback'):
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

def process_transcription_async(filepath, filename, session_id):
    """Process transcription in background thread with progress updates"""
    try:
        # Stage 1: Analyze audio
        update_progress(session_id, 'analyze', 25, 'Analyzing audio...', 'Checking file format and audio quality')
        
        audio_info = transcription_service.get_audio_info(filepath)
        duration_str = f"{audio_info['duration']:.1f}s"
        file_size = os.path.getsize(filepath)
        file_size_str = f"{file_size/1024/1024:.1f}MB"
        
        # Check if audio has meaningful content
        if not audio_info['has_audio']:
            update_progress(session_id, 'error', 0, 'Error', 'No audio content detected in file')
            return
        
        # Stage 2: Processing
        update_progress(session_id, 'process', 45, 'Processing audio...', 'Converting and optimizing for transcription')
        time.sleep(0.5)  # Small delay to show progress
        
        # Determine processing method based on file size and duration
        max_simple_size = 5 * 1024 * 1024  # 5MB threshold
        max_simple_duration = 60  # 60 seconds threshold
        
        # Stage 3: Transcription with real-time chunks
        update_progress(session_id, 'transcribe', 70, 'Transcribing speech...', 'AI is converting speech to text')
        
        # Initialize partial transcription
        progress_data[session_id]['partial_transcription'] = ""
        
        try:
            if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                # For small files, simulate chunked progress
                transcription = transcribe_with_progress(transcription_service.transcribe_simple, filepath, session_id, 70, 85)
                method = "Direct processing (small file)"
            else:
                # Use actual chunked transcription
                transcription = transcribe_with_progress(transcription_service.transcribe, filepath, session_id, 70, 85)
                method = "Chunked processing"
        except Exception as transcribe_error:
            # Fallback method
            try:
                if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                    transcription = transcribe_with_progress(transcription_service.transcribe, filepath, session_id, 70, 85)
                    method = "Chunked processing (fallback)"
                else:
                    transcription = transcribe_with_progress(transcription_service.transcribe_simple, filepath, session_id, 70, 85)
                    method = "Direct processing (fallback)"
            except Exception as fallback_error:
                try:
                    processed_path = transcription_service.preprocess_audio(filepath)
                    transcription = transcribe_with_progress(transcription_service.transcribe, processed_path, session_id, 70, 85)
                    method = "Preprocessed + chunked"
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    raise transcribe_error
        
        # Stage 4: Summarization
        update_progress(session_id, 'summarize', 90, 'Generating summary...', 'Creating intelligent summary of content')
        
        summary = summarizer.summarize(transcription) if transcription else ""
        
        # Check if transcription is meaningful
        if not transcription or transcription.lower() in ["no speech detected.", "no clear speech detected in the audio file.", ""]:
            update_progress(session_id, 'error', 0, 'Error', 'No speech could be detected in the audio file')
            return
        
        # Stage 5: Complete
        result = {
            'transcription': transcription,
            'summary': summary,
            'filename': filename,
            'duration': duration_str,
            'method': method,
            'file_size': file_size_str,
            'audio_quality': "Good" if audio_info['max_amplitude'] > 0.1 else "Low volume detected"
        }
        
        update_progress(session_id, 'complete', 100, 'Complete!', 'Transcription finished successfully')
        progress_data[session_id]['result'] = result
        
    except Exception as e:
        error_msg = str(e)
        # Provide more user-friendly error messages
        if "No such file" in error_msg or "cannot identify image file" in error_msg:
            error_msg = "File upload failed or file is corrupted. Please try again with a different file."
        elif "not supported" in error_msg.lower() or "format" in error_msg.lower():
            error_msg = "Audio format not supported or file is corrupted. Please convert to MP3, WAV, or M4A."
        elif "memory" in error_msg.lower():
            error_msg = "File too large or system out of memory. Please try a smaller file or shorter audio clip."
        elif "Model not properly loaded" in error_msg:
            error_msg = "Transcription service not available. Please try again later."
        elif "No audio content" in error_msg:
            error_msg = "The uploaded file doesn't contain detectable audio. Please check your file and try again."
        
        update_progress(session_id, 'error', 0, 'Error', f"Transcription failed: {error_msg}")
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# ----------------------
# Routes
# ----------------------

# Home page
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")

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
        return redirect(url_for('index'))
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

# Start transcription with file upload
@app.route("/transcribe", methods=["POST"])
@login_required
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file format. Please use: {', '.join(allowed_extensions)}"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save uploaded file
        file.save(filepath)
        
        # Generate unique session ID for this transcription
        session_id = str(uuid.uuid4())
        
        # Initialize progress
        update_progress(session_id, 'upload', 15, 'File uploaded successfully', 'Starting transcription process')
        
        # Start background processing
        thread = threading.Thread(
            target=process_transcription_async,
            args=(filepath, filename, session_id)
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
    app.run(debug=True, threaded=True)  # Enable threading for SSE