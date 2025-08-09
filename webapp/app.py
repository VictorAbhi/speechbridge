from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from auth import AuthHandler
from summarizer import SummarizerService
from transcription import TranscriptionService
from functools import wraps
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-change-this"  # change to a secure random key
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
auth_handler = AuthHandler()
transcription_service = TranscriptionService()
summarizer = SummarizerService()

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

# ----------------------
# Routes
# ----------------------

# Home page
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")  # Changed to index.html

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

# Transcription API (protected)
@app.route("/transcribe", methods=["POST"])
@login_required
def transcribe():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")
    
    file = request.files["file"]
    if file.filename == '':
        return render_template("index.html", error="No file selected")

    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return render_template("index.html", error=f"Unsupported file format. Please use: {', '.join(allowed_extensions)}")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save uploaded file
        file.save(filepath)
        
        # Get audio info first
        try:
            audio_info = transcription_service.get_audio_info(filepath)
            duration_str = f"{audio_info['duration']:.1f}s"
            file_size = os.path.getsize(filepath)
            file_size_str = f"{file_size/1024/1024:.1f}MB"
            
            # Check if audio has meaningful content
            if not audio_info['has_audio']:
                return render_template("index.html", error="No audio content detected in file. The file may be corrupted or contain only silence.")
                
        except Exception as info_error:
            return render_template("index.html", error=f"Could not read audio file: {str(info_error)}")
        
        # Determine processing method based on file size and duration
        max_simple_size = 5 * 1024 * 1024  # 5MB threshold
        max_simple_duration = 60  # 60 seconds threshold
        
        try:
            if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                # Try simple transcription for small files
                print(f"🎵 Using simple transcription for {filename}")
                transcription = transcription_service.transcribe_simple(filepath)
                method = "Direct processing (small file)"
            else:
                # Use chunked transcription for larger files
                print(f"🎵 Using chunked transcription for {filename}")
                transcription = transcription_service.transcribe(filepath)
                method = "Chunked processing"
                
            # Generate summary if transcription is successful
            summary = summarizer.summarize(transcription) if transcription else ""
                
        except Exception as transcribe_error:
            # If the preferred method fails, try the other one
            try:
                print(f"⚠️ Primary method failed, trying fallback for {filename}")
                if file_size < max_simple_size and audio_info['duration'] < max_simple_duration:
                    transcription = transcription_service.transcribe(filepath)
                    method = "Chunked processing (fallback)"
                else:
                    transcription = transcription_service.transcribe_simple(filepath)
                    method = "Direct processing (fallback)"
                # Generate summary for fallback transcription
                summary = summarizer.summarize(transcription) if transcription else ""
            except Exception as fallback_error:
                # If both methods fail, try with preprocessing
                try:
                    print(f"🔧 Trying with audio preprocessing for {filename}")
                    processed_path = transcription_service.preprocess_audio(filepath)
                    transcription = transcription_service.transcribe(processed_path)
                    method = "Preprocessed + chunked"
                    summary = summarizer.summarize(transcription) if transcription else ""
                    # Clean up processed file
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    raise transcribe_error  # Return the original error
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Check if transcription is meaningful
        if not transcription or transcription.lower() in ["no speech detected.", "no clear speech detected in the audio file.", ""]:
            return render_template("index.html", 
                                 error="No speech could be detected in the audio file. Please check that the file contains clear speech and try again.")
        
        return render_template("index.html", 
                             transcription=transcription, 
                             summary=summary,
                             filename=filename,
                             duration=duration_str,
                             method=method,
                             file_size=file_size_str,
                             audio_quality="Good" if audio_info['max_amplitude'] > 0.1 else "Low volume detected")
        
    except Exception as e:
        # Clean up uploaded file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        
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
        
        return render_template("index.html", error=f"Transcription failed: {error_msg}")

# Remove separate summarize route since it's now integrated with transcribe
# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)