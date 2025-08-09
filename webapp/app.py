from flask import Flask, jsonify, request, render_template
from auth_handler import auth_handler
from transcription_handler import transcription_handler
from functools import wraps
import jwt

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"  # change to a secure random key

# ----------------------
# JWT Decorator
# ----------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return jsonify({"message": "Token is missing"}), 401
        try:
            jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except Exception:
            return jsonify({"message": "Token is invalid"}), 401
        return f(*args, **kwargs)
    return decorated

# ----------------------
# Routes
# ----------------------

# GET login page
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

# POST login credentials
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"message": "Missing username or password"}), 400
    return auth_handler(data["username"], data["password"])

# Home page (protected)
@app.route("/", methods=["GET"])
@token_required
def index():
    return render_template("index.html")

# Transcription API (protected)
@app.route("/transcribe", methods=["POST"])
@token_required
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    return transcription_handler(file)

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
