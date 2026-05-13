# ============================================================
# app.py  —  Flask Object Detection & Counting Application
# ============================================================
import os, json
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from detector import DetectionEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

for d in ("uploads","models","logs"):
    os.makedirs(d, exist_ok=True)

engine = DetectionEngine()

def _ext(fn): return fn.rsplit(".",1)[-1].lower() if "." in fn else ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(engine.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/load_model", methods=["POST"])
def load_model():
    d = request.get_json()
    return jsonify(engine.load_model(d.get("model_type","pretrained"),
                                     d.get("model_path","")))

@app.route("/api/start", methods=["POST"])
def start(): return jsonify(engine.start(request.get_json()))

@app.route("/api/stop", methods=["POST"])
def stop():  return jsonify(engine.stop())

@app.route("/api/stats")
def stats(): return jsonify(engine.get_stats())

@app.route("/api/sessions")
def sessions(): return jsonify(engine.get_sessions())

@app.route("/api/clear_sessions", methods=["POST"])
def clear_sessions():
    p = os.path.join("logs","sessions.json")
    if os.path.exists(p): os.remove(p)
    return jsonify({"success": True})

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    if not f or not f.filename: return jsonify({"success":False,"message":"No file"})
    if _ext(f.filename) not in {"mp4","avi","mov","mkv","webm"}:
        return jsonify({"success":False,"message":"Invalid format"})
    name = secure_filename(f.filename)
    path = os.path.join("uploads", name)
    f.save(path)
    return jsonify({"success":True,"path":path,"name":name})

@app.route("/api/upload_model", methods=["POST"])
def upload_model():
    f = request.files.get("file")
    if not f or _ext(f.filename) != "pt":
        return jsonify({"success":False,"message":"Only .pt files"})
    name = secure_filename(f.filename)
    path = os.path.join("models", name)
    f.save(path)
    return jsonify({"success":True,"path":path,"name":name})

@app.route("/api/list_models")
def list_models():
    return jsonify([f for f in os.listdir("models") if f.endswith(".pt")])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
