import os
import time
import base64
import urllib.request
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, Response, request as flask_request
from flask_socketio import SocketIO, emit
from detector import YOLODetector
from tracker import ByteTracker, Track
from utils.person_state import PersonStateManager
from utils.reid import AppearanceGallery

app = Flask(__name__)
app.config["SECRET_KEY"] = "ai-crowd-counter-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

detector = None
gallery = AppearanceGallery(data_file="data/reid_gallery.json")
Track.seed_next_id(gallery.get_max_id())
person_manager = PersonStateManager()
tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=150, gallery=gallery, min_hits=2)


def init_detector():
    global detector
    model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    detector = YOLODetector(model_path=model_path, confidence=0.5)
    print("[INFO] YOLOv8 detector initialized.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "active",
        "detector_ready": detector is not None,
        "active_count": len(person_manager.get_active_persons()),
        "current_count": person_manager.get_current_count(),
    })


@app.route("/api/snapshot")
def proxy_snapshot():
    url = flask_request.args.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            img_bytes = resp.read()
        return Response(img_bytes, mimetype="image/jpeg", headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@socketio.on("connect")
def handle_connect():
    print("[INFO] Client connected.")
    emit("status", {"message": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("[INFO] Client disconnected.")


@socketio.on("video_frame")
def handle_frame(data):
    try:
        img_data = base64.b64decode(data["frame"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            emit("error", {"message": "Failed to decode frame"})
            return

        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame=frame)
        person_manager.update(tracked)

        for person in tracked:
            x1, y1, x2, y2 = [int(c) for c in person["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 64, 175), 2)
            label = f"ID: {person['id']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 4), (30, 64, 175), -1)
            cv2.putText(frame, label, (x1 + 3, label_y - 2), font, font_scale, (255, 255, 255), thickness)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_frame = base64.b64encode(buffer).decode("utf-8")

        emit("processed_frame", {
            "frame": "data:image/jpeg;base64," + encoded_frame,
            "current_count": person_manager.get_current_count(),
            "live_persons": person_manager.get_live_persons(),
            "active_persons": person_manager.get_active_persons(),
            "all_persons": person_manager.get_all_persons(),
        })
    except Exception as e:
        print(f"[ERROR] {e}")
        emit("error", {"message": str(e)})


if __name__ == "__main__":
    init_detector()
    print(f"[INFO] Known persons in gallery: {gallery.get_known_count()}")
    print("[INFO] Starting AI Crowd Counter on http://0.0.0.0:5000")
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        person_manager.save_now()
        gallery.save_now()
        print("[INFO] Data saved.")
