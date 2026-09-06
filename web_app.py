import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame
from flask import Flask, Response, render_template_string, jsonify
import numpy as np
import threading
import time

app = Flask(__name__)

# Initialize Pygame Mixer for sound alerts
try:
    pygame.mixer.init()
    alarm_sleep = pygame.mixer.Sound("alarm.mp3")
    alarm_facehide = pygame.mixer.Sound("faudio.mp3")
    alarm_phone = pygame.mixer.Sound("paudio.mp3")
    audio_enabled = True
except Exception as e:
    print(f"Audio init error: {e}")
    audio_enabled = False
    alarm_sleep = alarm_facehide = alarm_phone = None

# Track active audio state
current_playing = None

# Landmark Indices for Sleep Detection
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

# Frame Counters & Thresholds
closed_frames = 0
SLEEP_THRESHOLD_FRAMES = 15

covered_frames = 0
COVER_THRESHOLD_FRAMES = 20

# Global state for frontend dashboard telemetry
telemetry = {
    "eye_ratio": 0,
    "status": "Initializing...",
    "face_detected": False,
    "is_sleepy": False,
    "is_face_covered": False,
    "phone_detected": False,
    "current_alert": "Normal"
}

# Initialize Camera, Face Mesh, and YOLO Model
cap = None
face_detector = None
phone_detector = None
classNames = {}

def init_detectors():
    global cap, face_detector, phone_detector, classNames
    try:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
    except Exception as e:
        print(f"Camera init error: {e}")

    try:
        if face_detector is None:
            face_detector = FaceMeshDetector(maxFaces=1)
    except Exception as e:
        print(f"Face detector init error: {e}")

    try:
        if phone_detector is None:
            phone_detector = YOLO("yolov8n.pt")
            classNames = phone_detector.names
    except Exception as e:
        print(f"YOLO detector init error: {e}")

def generate_frames():
    global cap, face_detector, phone_detector, classNames
    global closed_frames, covered_frames, current_playing, telemetry

    init_detectors()

    while True:
        if cap is None or not cap.isOpened():
            # Generate black placeholder frame with message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for Camera / Model...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        is_audio_busy = pygame.mixer.get_busy() if audio_enabled else False
        if not is_audio_busy:
            current_playing = None

        # 1. SLEEP & FACE COVER DETECTION
        is_sleepy = False
        is_face_covered = False
        ratio = 0

        if face_detector:
            try:
                img, faces = face_detector.findFaceMesh(img, draw=False)
            except Exception:
                faces = []

            if faces:
                covered_frames = 0
                face = faces[0]
                eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
                face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])

                if face_dist > 0:
                    ratio = (eye_dist / face_dist) * 100

                if ratio < 11.0:
                    closed_frames += 1
                else:
                    closed_frames = 0

                if closed_frames >= SLEEP_THRESHOLD_FRAMES:
                    is_sleepy = True

                cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1)
                telemetry["face_detected"] = True
            else:
                closed_frames = 0
                covered_frames += 1
                if covered_frames >= COVER_THRESHOLD_FRAMES:
                    is_face_covered = True
                telemetry["face_detected"] = False
        
        telemetry["eye_ratio"] = int(ratio)
        telemetry["is_sleepy"] = is_sleepy
        telemetry["is_face_covered"] = is_face_covered

        # 2. PHONE DETECTION
        phone_detected = False
        if phone_detector:
            try:
                results = phone_detector.predict(img, stream=True, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if classNames.get(cls_id) == "cell phone" and conf > 0.5:
                            phone_detected = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cvzone.putTextRect(img, f"Phone detected! {int(conf*100)}%",
                                               (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))
            except Exception as e:
                pass

        telemetry["phone_detected"] = phone_detected

        # 3. ALARM LOGIC & DISPLAY
        alert_msg = "Studying Normally"
        alert_color = (0, 255, 0)

        if is_face_covered:
            alert_msg = "DONT COVER YOUR FACE!"
            cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
            if audio_enabled and not is_audio_busy and alarm_facehide:
                alarm_facehide.play(0)
                current_playing = 'facehide'
        elif is_sleepy:
            alert_msg = "WAKE UP & STUDY!"
            cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
            if audio_enabled and not is_audio_busy and alarm_sleep:
                alarm_sleep.play(0)
                current_playing = 'sleep'
        elif phone_detected:
            alert_msg = "PUT THE PHONE AWAY!"
            cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))
            if audio_enabled and not is_audio_busy and alarm_phone:
                alarm_phone.play(0)
                current_playing = 'phone'
        elif is_audio_busy:
            if current_playing == 'facehide':
                alert_msg = "DONT COVER YOUR FACE!"
                cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
            elif current_playing == 'sleep':
                alert_msg = "WAKE UP & STUDY!"
                cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
            elif current_playing == 'phone':
                alert_msg = "PUT THE PHONE AWAY!"
                cvzone.putTextRect(img, alert_msg, (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))

        telemetry["current_alert"] = alert_msg

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Study Monitor - Live Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(23, 32, 54, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(239, 68, 68, 0.1) 0px, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.25rem 2rem;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-weight: 600;
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.9); }
        }

        .container {
            max-width: 1380px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.5rem;
            flex: 1;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .video-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 460px;
            border-radius: 1.25rem 1.25rem 0 0;
            overflow: hidden;
        }

        .video-container img {
            width: 100%;
            height: auto;
            max-height: 580px;
            object-fit: contain;
            display: block;
        }

        .video-footer {
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.6);
            border-top: 1px solid var(--card-border);
        }

        .alert-banner {
            padding: 0.6rem 1.2rem;
            border-radius: 0.75rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        .alert-normal {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .alert-danger {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
            animation: shake 0.5s ease;
        }

        .alert-warning {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20%, 60% { transform: translateX(-5px); }
            40%, 80% { transform: translateX(5px); }
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .metric-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--card-border);
            border-radius: 0.85rem;
            padding: 1rem;
            transition: transform 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
        }

        .metric-label {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .status-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }

        .status-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1rem;
            background: rgba(15, 23, 42, 0.4);
            border-radius: 0.75rem;
            border: 1px solid var(--card-border);
        }

        .status-name {
            font-size: 0.9rem;
            font-weight: 500;
        }

        .tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 0.5rem;
        }

        .tag-ok {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }

        .tag-alert {
            background: rgba(239, 68, 68, 0.25);
            color: #f87171;
        }

        .tag-warn {
            background: rgba(245, 158, 11, 0.25);
            color: #fbbf24;
        }

        .rules-card {
            font-size: 0.85rem;
            line-height: 1.5;
            color: var(--text-muted);
        }

        .rules-card li {
            margin-left: 1.25rem;
            margin-bottom: 0.4rem;
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <span>🎯</span> Smart Study Monitor AI
        </div>
        <div class="badge">
            <span class="badge-dot"></span> Live Localhost Monitor
        </div>
    </header>

    <div class="container">
        <!-- Live Video Player Card -->
        <div class="video-card">
            <div class="video-container">
                <img id="liveStream" src="/video_feed" alt="Webcam Live Video Feed" onerror="this.src='/placeholder'">
            </div>
            <div class="video-footer">
                <div id="alertBanner" class="alert-banner alert-normal">
                    Checking Activity...
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">
                    Webcam Active (640x480)
                </div>
            </div>
        </div>

        <!-- Telemetry & Stats Sidebar -->
        <div class="sidebar">
            <div class="panel">
                <div class="panel-title">
                    <span>📊</span> Real-Time Telemetry
                </div>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Eye Ratio</div>
                        <div id="eyeRatioVal" class="metric-value">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Face Mesh</div>
                        <div id="faceMeshVal" class="metric-value" style="font-size: 1.1rem; padding-top: 4px;">--</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">
                    <span>🛡️</span> Detection Triggers
                </div>
                <div class="status-list">
                    <div class="status-item">
                        <span class="status-name">😴 Sleep / Drowsiness</span>
                        <span id="sleepTag" class="tag tag-ok">Awake</span>
                    </div>
                    <div class="status-item">
                        <span class="status-name">🙈 Face Covered</span>
                        <span id="faceTag" class="tag tag-ok">Visible</span>
                    </div>
                    <div class="status-item">
                        <span class="status-name">📱 Cell Phone</span>
                        <span id="phoneTag" class="tag tag-ok">None</span>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">
                    <span>ℹ️</span> Active Rules
                </div>
                <ul class="rules-card">
                    <li><strong>Eye Distance Ratio:</strong> Warning triggers if ratio &lt; 11% for 15+ frames.</li>
                    <li><strong>Covered Face:</strong> Alert triggers if face is missing/hidden for 20+ frames.</li>
                    <li><strong>YOLOv8 Phone:</strong> Real-time detection with &gt;50% confidence.</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();

                document.getElementById('eyeRatioVal').textContent = data.eye_ratio;
                document.getElementById('faceMeshVal').textContent = data.face_detected ? 'Tracked' : 'Searching';
                document.getElementById('faceMeshVal').style.color = data.face_detected ? '#34d399' : '#f59e0b';

                // Sleep Tag
                const sleepTag = document.getElementById('sleepTag');
                if (data.is_sleepy) {
                    sleepTag.className = 'tag tag-alert';
                    sleepTag.textContent = 'SLEEPING!';
                } else {
                    sleepTag.className = 'tag tag-ok';
                    sleepTag.textContent = 'Awake';
                }

                // Face Tag
                const faceTag = document.getElementById('faceTag');
                if (data.is_face_covered) {
                    faceTag.className = 'tag tag-alert';
                    faceTag.textContent = 'COVERED!';
                } else {
                    faceTag.className = 'tag tag-ok';
                    faceTag.textContent = 'Visible';
                }

                // Phone Tag
                const phoneTag = document.getElementById('phoneTag');
                if (data.phone_detected) {
                    phoneTag.className = 'tag tag-warn';
                    phoneTag.textContent = 'DETECTED!';
                } else {
                    phoneTag.className = 'tag tag-ok';
                    phoneTag.textContent = 'Clear';
                }

                // Banner
                const banner = document.getElementById('alertBanner');
                banner.textContent = data.current_alert;
                if (data.current_alert === 'WAKE UP & STUDY!' || data.current_alert === 'DONT COVER YOUR FACE!') {
                    banner.className = 'alert-banner alert-danger';
                } else if (data.current_alert === 'PUT THE PHONE AWAY!') {
                    banner.className = 'alert-banner alert-warning';
                } else {
                    banner.className = 'alert-banner alert-normal';
                }

            } catch (err) {
                console.error("Telemetry fetch error:", err);
            }
        }

        setInterval(fetchTelemetry, 350);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(telemetry)

if __name__ == '__main__':
    print("Starting Smart Study Monitor Web Server on http://localhost:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
