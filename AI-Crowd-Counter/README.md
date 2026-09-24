# AI Crowd Counter

A full-stack real-time people detection, counting, and tracking system with a professional surveillance dashboard.

## Features

- Real-time people detection via YOLOv8
- Unique Person ID assignment and persistent tracking (ByteTrack)
- Blue bounding boxes with IDs displayed above each person
- Mobile phone camera support (WebRTC/getUserMedia)
- 5-minute minimum duration filter for tracked individuals
- Live crowd counter
- Dark Indigo Blue glassmorphism UI
- Responsive Bootstrap 5 dashboard
- Real-time updates via WebSocket (Flask-SocketIO)

## Requirements

- Python 3.9+
- pip
- A webcam or mobile phone camera accessible via browser

## Installation

```bash
cd AI-Crowd-Counter
pip install -r requirements.txt
```

The YOLOv8n model will be downloaded automatically on first run.

## Usage

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Click **Start Camera** and grant camera access.

## Architecture

```
AI-Crowd-Counter/
├── app.py                  # Flask + SocketIO server
├── requirements.txt
├── models/                 # YOLO weights (auto-downloaded)
├── detector/
│   └── yolo_detector.py   # YOLOv8 person detection
├── tracker/
│   └── byte_tracker.py    # ByteTrack multi-object tracker
├── utils/
│   └── person_state.py    # Person state & duration management
├── templates/
│   └── index.html         # Dashboard HTML
├── static/
│   ├── css/style.css      # Dark Indigo theme styles
│   └── js/main.js         # Frontend camera + SocketIO logic
└── README.md
```

## How It Works

1. Browser captures camera frames and sends them to the server via WebSocket
2. Server runs YOLOv8 detection on each frame
3. ByteTrack assigns and maintains unique Person IDs across frames
4. Annotated frames (bounding boxes + IDs) are sent back to the browser
5. Person durations are tracked server-side; persons visible for 5+ minutes appear in the tracking panel
