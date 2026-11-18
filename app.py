import cv2
import math
import json
import csv
import numpy as np
import threading
import os
import time
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template_string, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR")
if not DATA_DIR:
    if os.path.exists("/app/data"):
        DATA_DIR = "/app/data"
    else:
        DATA_DIR = os.path.join(BASE_DIR, "data")

ZONES_FILE = os.path.join(DATA_DIR, "zones.json")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
DB_FILE = os.path.join(DATA_DIR, "alerts.db")

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                violated_zones TEXT,
                image_file TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

def save_alert_to_db(timestamp_str, alert_type, violated_zones_str, image_filename):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, alert_type, violated_zones, image_file)
                VALUES (?, ?, ?, ?)
            """, (timestamp_str, alert_type, violated_zones_str, image_filename))
            conn.commit()
    except Exception as e:
        print(f"[DB] Lỗi lưu cảnh báo vào database: {e}")

def load_saved_zones():
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[Zones] Lỗi đọc file zones.json: {e}")
    return [[[0, 0], [320, 0], [320, 240], [0, 240]]]

def save_zones_to_disk(zones):
    try:
        os.makedirs(os.path.dirname(ZONES_FILE), exist_ok=True)
        with open(ZONES_FILE, "w", encoding="utf-8") as f:
            json.dump(zones, f)
    except Exception as e:
        print(f"[Zones] Lỗi lưu file zones.json: {e}")

print("Đang nạp mô hình YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt')

zones_list = load_saved_zones()
violation_streak = 0
fall_streak = 0
lock = threading.Lock()
alert_state_lock = threading.Lock()
last_logged_alert_type = "none"
last_alert_snapshot_attempt = 0.0

def save_alert_snapshot(frame, alert_type, violated_indices):
    os.makedirs(ALERTS_DIR, exist_ok=True)
    timestamp = datetime.now().astimezone()
    timestamp_iso = timestamp.isoformat(timespec="seconds")
    image_filename = f"{alert_type}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    image_path = os.path.join(ALERTS_DIR, image_filename)

    if not cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Không thể lưu ảnh cảnh báo: {image_path}")

    violated_zones_str = ";".join(str(index + 1) for index in violated_indices)

    # 1. Lưu log vào SQLite Database
    save_alert_to_db(timestamp_iso, alert_type, violated_zones_str, image_filename)

    # 2. Ghi dự phòng vào file CSV
    log_path = os.path.join(ALERTS_DIR, "alerts.csv")
    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=["timestamp", "alert_type", "violated_zones", "image_file"]
        )
        if log_file.tell() == 0:
            log_writer.writeheader()
        log_writer.writerow({
            "timestamp": timestamp_iso,
            "alert_type": alert_type,
            "violated_zones": violated_zones_str,
            "image_file": image_filename
        })

    return image_filename

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def is_real_human(kpts_xy, kpts_conf):
    if kpts_conf is None or len(kpts_conf) < 13:
        return False
    valid_joints = sum(1 for c in kpts_conf if c > 0.5)
    return valid_joints >= 5

def check_fall_by_skeleton(keypoints, kpts_conf, box):
    if len(keypoints) < 13 or kpts_conf is None:
        return False

    x1, y1, x2, y2 = box[:4]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    has_shoulders = (kpts_conf[5] > 0.5 and kpts_conf[6] > 0.5 and left_shoulder[0] > 0 and right_shoulder[0] > 0)
    has_hips = (kpts_conf[11] > 0.5 and kpts_conf[12] > 0.5 and left_hip[0] > 0 and right_hip[0] > 0)

    # 1. Khi camera nhìn thấy toàn thân (đủ cả vai và hông)
    if has_shoulders and has_hips:
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hip_y = (left_hip[1] + right_hip[1]) / 2.0
        shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        hip_x = (left_hip[0] + right_hip[0]) / 2.0

        dx = abs(shoulder_x - hip_x)
        dy = abs(shoulder_y - hip_y)
        angle = math.degrees(math.atan2(dx, dy + 1e-5))

        # Thân người đổ nghiêng > 58 độ mới tính ngã
        if angle > 58:
            return True

        # Đầu hạ thấp ngang bằng hoặc dưới hông
        if kpts_conf[0] > 0.5 and nose[1] > 0 and nose[1] > (hip_y - 10):
            return True

        # Chỉ kiểm tra khung nằm ngang khi thấy đủ cả hông và chiều rộng gấp > 1.45 lần chiều cao
        if box_w > (box_h * 1.45):
            return True

    # 2. Khi ngồi gần webcam (chỉ thấy phần thân trên, khuất hông)
    elif has_shoulders:
        s_dx = abs(left_shoulder[0] - right_shoulder[0])
        s_dy = abs(left_shoulder[1] - right_shoulder[1])
        shoulder_tilt = math.degrees(math.atan2(s_dy, s_dx + 1e-5))

        # Hai vai nghiêng dốc hẳn > 55 độ (đổ gục sang trái/phải)
        if shoulder_tilt > 55:
            return True

        # Đầu gục hẳn xuống dưới đường vai một khoảng lớn (> 45px) hoặc tụt sát đáy khung hình
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        if kpts_conf[0] > 0.5 and nose[1] > 0:
            if nose[1] > (shoulder_y + 45) or nose[1] > 420:
                return True

    return False

@app.route('/get_zones', methods=['GET'])
def get_zones():
    with lock:
        return jsonify({"zones": zones_list})

@app.route('/update_zones', methods=['POST'])
def update_zones():
    global zones_list
    data = request.get_json(silent=True)
    if not data or ('points' not in data and 'zones' not in data):
        return jsonify({"error": "Invalid data"}), 400

    raw_zones = data.get('zones', [data.get('points', [])])
    valid_zones = [z for z in raw_zones if len(z) >= 3]
    with lock:
        zones_list = valid_zones
        save_zones_to_disk(zones_list)
    return jsonify({"status": "updated", "count": len(zones_list)})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global violation_streak, fall_streak
    global last_logged_alert_type, last_alert_snapshot_attempt
    file = request.files.get('frame')
    if not file:
        return jsonify({"error": "No frame"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))

    results = model(img, conf=0.55, classes=[0], imgsz=480, verbose=False)

    raw_violation = False
    raw_fall = False
    detections_out = []
    violated_zone_indices = set()

    with lock:
        current_polygons = [np.array(z, np.int32) for z in zones_list]

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints_list = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
        keypoints_conf = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else []

        for i, box in enumerate(boxes):
            kpts = keypoints_list[i].tolist() if i < len(keypoints_list) else []
            kpts_c = keypoints_conf[i].tolist() if i < len(keypoints_conf) else []

            if not is_real_human(kpts, kpts_c):
                continue

            x1, y1, x2, y2 = box[:4]
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            in_any_zone = False
            person_zones = []
            for z_idx, poly in enumerate(current_polygons):
                if is_point_in_polygon(center_point, poly):
                    in_any_zone = True
                    raw_violation = True
                    violated_zone_indices.add(z_idx)
                    person_zones.append(z_idx + 1)

            is_falling = bool(check_fall_by_skeleton(kpts, kpts_c, box))
            if is_falling:
                raw_fall = True

            detections_out.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "in_zone": in_any_zone,
                "violated_zones": person_zones,
                "is_falling": is_falling,
                "keypoints": kpts
            })

    with alert_state_lock:
        violation_streak = (violation_streak + 1) if raw_violation else 0
        fall_streak = (fall_streak + 1) if raw_fall else 0

        has_violation = violation_streak >= 3
        has_fall = fall_streak >= 4

        alert_type = "none"
        if has_fall:
            alert_type = "fall"
        elif has_violation:
            alert_type = "violation"

        snapshot_saved = False
        snapshot_file = None
        if alert_type == "none":
            last_logged_alert_type = "none"
            last_alert_snapshot_attempt = 0.0
        else:
            now = time.monotonic()
            if (alert_type != last_logged_alert_type or
                    now - last_alert_snapshot_attempt >= 10):
                last_logged_alert_type = alert_type
                last_alert_snapshot_attempt = now
                try:
                    snapshot_file = save_alert_snapshot(
                        img, alert_type, violated_zone_indices
                    )
                    snapshot_saved = True
                except (OSError, cv2.error):
                    app.logger.exception("Lưu ảnh/log cảnh báo thất bại")

    return jsonify({
        "detections": detections_out,
        "has_violation": has_violation,
        "has_fall": has_fall,
        "violated_indices": list(violated_zone_indices) if has_violation else [],
        "alert_type": alert_type,
        "snapshot_saved": snapshot_saved,
        "snapshot_file": snapshot_file
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Camera Giám Sát Đa Vùng Cấm</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #0f1115;
            color: white;
            margin: 0;
            padding: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            user-select: none;
        }
        h2 { margin: 4px 0 10px 0; font-size: 20px; }
        .controls {
            margin-bottom: 8px;
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 9px 14px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 13px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        button.fullscreen { background: #6a1b9a; }
        button.full { background: #1565c0; }
        button.undo { background: #f57c00; }
        button.clear { background: #c62828; }
        button.reset { background: #37474f; }
        button.history { background: #00796b; }
        .badge {
            padding: 8px 12px;
            border-radius: 5px;
            background: #263238;
            font-size: 13px;
            border: 1px solid #455a64;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.75);
            overflow-y: auto;
        }
        .modal-content {
            background-color: #1e222b;
            margin: 4% auto;
            padding: 18px;
            border-radius: 8px;
            width: 90%;
            max-width: 920px;
            border: 1px solid #444;
            color: #fff;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .close-btn {
            background: #d32f2f;
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            border: none;
        }
        table.alert-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        table.alert-table th, table.alert-table td {
            border: 1px solid #37474f;
            padding: 8px;
            text-align: left;
        }
        table.alert-table th {
            background: #263238;
        }
        table.alert-table tr:nth-child(even) {
            background: #171b22;
        }
        .thumb-img {
            max-width: 80px;
            max-height: 50px;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .thumb-img:hover {
            transform: scale(2.2);
        }
        .hint { color: #ffd54f; font-size: 13px; margin-bottom: 8px; font-weight: bold; }
        .video-container {
            width: 94vw;
            max-width: 1120px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        canvas {
            width: 100%;
            height: auto;
            max-height: 82vh;
            object-fit: contain;
            border: 3px solid #444;
            border-radius: 10px;
            background: #000;
            cursor: crosshair;
            display: block;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        }
    </style>
</head>
<body>
    <h2>🎥 Giám Sát Đa Vùng Cấm & An Toàn Lao Động (YOLOv8-Pose)</h2>
    <div class="controls">
        <button id="audioBtn" onclick="enableAudio()">🔊 Bật Âm Thanh & Còi</button>
        <button class="fullscreen" onclick="toggleFullscreen()">⛶ Toàn màn hình</button>
        <button class="full" onclick="setFullScreenZone()">🌐 Cấm toàn khung</button>
        <button class="undo" onclick="undoLastZone()">↩️ Xóa vùng cuối</button>
        <button class="clear" onclick="clearAllZones()">🗑️ Xóa tất cả</button>
        <button class="reset" onclick="resetDefaultZone()">🔄 Mặc định</button>
        <button class="history" onclick="toggleAlertModal()">📋 Lịch Sử Cảnh Báo</button>
        <div id="audioBadge" class="badge">Âm thanh: Đang tắt</div>
        <div id="zoneBadge" class="badge">Số vùng cấm: 1</div>
    </div>
    <div class="hint">🖱️ Kéo chuột trái trên video để thêm vùng cấm • Bấm "⛶ Toàn màn hình" để phóng tối đa</div>
    <video id="video" width="640" height="480" autoplay playsinline style="display:none;"></video>
    <div class="video-container" id="videoContainer">
        <canvas id="canvas" width="640" height="480"></canvas>
    </div>

    <!-- Modal Lịch sử Cảnh báo Database -->
    <div id="historyModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 style="margin:0;">📋 Lịch Sử Cảnh Báo (Cơ Sở Dữ Liệu SQLite)</h3>
                <div>
                    <button onclick="fetchAlertHistory()" style="background:#0288d1; margin-right:8px;">🔄 Làm mới</button>
                    <button class="close-btn" onclick="toggleAlertModal()">✖ Đóng</button>
                </div>
            </div>
            <div style="overflow-x: auto; max-height: 60vh;">
                <table class="alert-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Thời gian</th>
                            <th>Loại cảnh báo</th>
                            <th>Vùng vi phạm</th>
                            <th>Hình ảnh snapshot</th>
                        </tr>
                    </thead>
                    <tbody id="alertTableBody">
                        <tr><td colspan="5" style="text-align:center;">Đang tải dữ liệu...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const zoneBadge = document.getElementById('zoneBadge');

        const offscreen = document.createElement('canvas');
        offscreen.width = 640;
        offscreen.height = 480;
        const offCtx = offscreen.getContext('2d');

        let audioEnabled = false;
        let audioCtx = null;
        let lastAlert = "none";
        let lastAlertTime = 0;
        let latestResult = { detections: [], has_violation: false, has_fall: false, violated_indices: [], alert_type: "none" };
        let isProcessing = false;

        let zonesList = [
            [[0, 0], [320, 0], [320, 240], [0, 240]]
        ];
        let isDrawing = false;
        let startX = 0, startY = 0;
        let currentX = 0, currentY = 0;

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                canvas.requestFullscreen().catch(err => alert("Không thể mở toàn màn hình: " + err.message));
            } else {
                document.exitFullscreen();
            }
        }

        function clamp(val, min, max) {
            return Math.max(min, Math.min(max, val));
        }

        function getMousePos(evt) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: clamp(Math.round((evt.clientX - rect.left) * scaleX), 0, canvas.width),
                y: clamp(Math.round((evt.clientY - rect.top) * scaleY), 0, canvas.height)
            };
        }

        canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            const pos = getMousePos(e);
            isDrawing = true;
            startX = pos.x;
            startY = pos.y;
            currentX = pos.x;
            currentY = pos.y;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const pos = getMousePos(e);
            currentX = pos.x;
            currentY = pos.y;
        });

        window.addEventListener('mouseup', async (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            const pos = getMousePos(e);
            const x1 = Math.min(startX, pos.x);
            const y1 = Math.min(startY, pos.y);
            const x2 = Math.max(startX, pos.x);
            const y2 = Math.max(startY, pos.y);

            if ((x2 - x1) > 15 && (y2 - y1) > 15) {
                const newPoly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
                zonesList.push(newPoly);
                await syncZonesToServer();
            }
        });

        async function syncZonesToServer() {
            try {
                await fetch('/update_zones', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ zones: zonesList })
                });
                zoneBadge.textContent = `Số vùng cấm: ${zonesList.length}`;
                zoneBadge.style.borderColor = "#4CAF50";
            } catch (err) {
                zoneBadge.textContent = "❌ Lỗi lưu vùng cấm";
            }
        }

        async function setFullScreenZone() {
            zonesList = [[[2, 2], [638, 2], [638, 478], [2, 478]]];
            await syncZonesToServer();
        }

        async function undoLastZone() {
            if (zonesList.length > 0) {
                zonesList.pop();
                await syncZonesToServer();
            }
        }

        async function clearAllZones() {
            zonesList = [];
            await syncZonesToServer();
        }

        async function resetDefaultZone() {
            zonesList = [[[0, 0], [320, 0], [320, 240], [0, 240]]];
            await syncZonesToServer();
        }

        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();

        function playAlarmBeep(isUrgent) {
            if (!audioEnabled || !audioCtx) return;
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = isUrgent ? 'sawtooth' : 'sine';
            osc.frequency.setValueAtTime(isUrgent ? 880 : 620, audioCtx.currentTime);
            if (isUrgent) osc.frequency.setValueAtTime(1100, audioCtx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.35);
        }

        function speakVietnamese(text) {
            if (!audioEnabled) return;
            try {
                if (window.speechSynthesis.speaking) return;
                const msg = new SpeechSynthesisUtterance(text);
                const voices = window.speechSynthesis.getVoices();
                const viVoice = voices.find(v => v.lang.toLowerCase().includes('vi'));
                if (viVoice) {
                    msg.voice = viVoice;
                    msg.lang = viVoice.lang;
                } else {
                    msg.lang = 'vi-VN';
                }
                msg.rate = 1.25;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            } catch (e) {}
        }

        function enableAudio() {
            audioEnabled = true;
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') audioCtx.resume();

            const btn = document.getElementById("audioBtn");
            const badge = document.getElementById("audioBadge");
            btn.style.backgroundColor = "#008CBA";
            btn.textContent = "🔊 Đã bật âm thanh";
            badge.style.backgroundColor = "#2e7d32";
            badge.textContent = "Âm thanh: ĐANG BẬT";

            playAlarmBeep(false);
            window.speechSynthesis.cancel();
            speakVietnamese("Hệ thống âm thanh cảnh báo đã sẵn sàng!");
        }

        async function startCamera() {
            try {
                const res = await fetch('/get_zones');
                const zdata = await res.json();
                if (zdata.zones) {
                    zonesList = zdata.zones;
                    zoneBadge.textContent = `Số vùng cấm: ${zonesList.length}`;
                }
            } catch (e) {}

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                await video.play();
                requestAnimationFrame(renderLoop);
                setInterval(sendFrameToAI, 150);
            } catch (err) {
                alert("Không mở được Webcam: " + err.message);
            }
        }

        function drawPolygon(pts, label, isViolated) {
            if (!pts || pts.length < 3) return;
            const strokeColor = isViolated ? "#FF1744" : "#00E676";
            const fillColor = isViolated ? "rgba(255, 23, 68, 0.28)" : "rgba(0, 230, 118, 0.14)";

            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
                ctx.lineTo(pts[i][0], pts[i][1]);
            }
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.fillStyle = strokeColor;
            ctx.font = "bold 13px Arial";
            ctx.fillText(label, pts[0][0] + 6, pts[0][1] + 18);
        }

        function renderLoop() {
            if (video.readyState >= 2) {
                ctx.drawImage(video, 0, 0, 640, 480);

                const violatedSet = new Set(latestResult.violated_indices || []);
                zonesList.forEach((poly, idx) => {
                    drawPolygon(poly, `VUNG #${idx + 1}`, violatedSet.has(idx));
                });

                if (isDrawing) {
                    ctx.save();
                    ctx.setLineDash([6, 4]);
                    ctx.strokeStyle = "#FFD600";
                    ctx.fillStyle = "rgba(255, 214, 0, 0.22)";
                    ctx.lineWidth = 2.5;
                    ctx.fillRect(startX, startY, currentX - startX, currentY - startY);
                    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
                    ctx.restore();
                }

                const edges = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
                for (const det of latestResult.detections) {
                    const [x1, y1, x2, y2] = det.box;
                    const isDanger = det.is_falling || det.in_zone;
                    ctx.strokeStyle = isDanger ? "#FF0000" : "#00FF00";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    ctx.fillStyle = "#FF0000";
                    ctx.font = "bold 15px Arial";
                    if (det.in_zone) {
                        const zText = det.violated_zones ? `XAM NHAP VUNG #${det.violated_zones.join(',#')}!` : "XAM NHAP!";
                        ctx.fillText(zText, x1, Math.max(20, y1 - 8));
                    }
                    if (det.is_falling) ctx.fillText("CO NGUOI NGA!", x1, Math.max(40, y1 - 28));

                    const kpts = det.keypoints;
                    if (kpts && kpts.length >= 17) {
                        ctx.strokeStyle = "#00FFFF";
                        ctx.lineWidth = 2;
                        for (const [i, j] of edges) {
                            if (kpts[i][0] > 0 && kpts[i][1] > 0 && kpts[j][0] > 0 && kpts[j][1] > 0) {
                                ctx.beginPath();
                                ctx.moveTo(kpts[i][0], kpts[i][1]);
                                ctx.lineTo(kpts[j][0], kpts[j][1]);
                                ctx.stroke();
                            }
                        }
                        ctx.fillStyle = "#FF0000";
                        for (const pt of kpts) {
                            if (pt[0] > 0 && pt[1] > 0) {
                                ctx.beginPath();
                                ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }
                    }
                }

                if (latestResult.has_fall) {
                    ctx.fillStyle = "#FF0000";
                    ctx.font = "bold 24px Arial";
                    ctx.fillText("FALL ALARM ACTIVE", 20, 45);
                } else if (latestResult.has_violation) {
                    ctx.fillStyle = "#FF0000";
                    ctx.font = "bold 24px Arial";
                    ctx.fillText("ZONE ALARM ACTIVE", 20, 45);
                }
            }
            requestAnimationFrame(renderLoop);
        }

        async function sendFrameToAI() {
            if (isProcessing || video.readyState < 2) return;
            isProcessing = true;
            offCtx.drawImage(video, 0, 0, 640, 480);
            offscreen.toBlob(async (blob) => {
                try {
                    const formData = new FormData();
                    formData.append('frame', blob, 'frame.jpg');
                    const res = await fetch('/process_frame', { method: 'POST', body: formData });
                    const data = await res.json();
                    latestResult = data;

                    const now = Date.now();
                    if (data.alert_type !== "none") {
                        if (data.alert_type !== lastAlert || (now - lastAlertTime > 3000)) {
                            if (data.alert_type === "fall") {
                                playAlarmBeep(true);
                                speakVietnamese("Cảnh báo nguy hiểm! Có người ngã!");
                            } else if (data.alert_type === "violation") {
                                playAlarmBeep(false);
                                speakVietnamese("Phát hiện xâm nhập vùng cấm!");
                            }
                            lastAlert = data.alert_type;
                            lastAlertTime = now;
                        }
                    } else {
                        lastAlert = "none";
                    }
                } catch (e) {} finally {
                    isProcessing = false;
                }
            }, 'image/jpeg', 0.6);
        }

        async function toggleAlertModal() {
            const modal = document.getElementById('historyModal');
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
            } else {
                modal.style.display = 'block';
                await fetchAlertHistory();
            }
        }

        async function fetchAlertHistory() {
            const tbody = document.getElementById('alertTableBody');
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Đang tải dữ liệu...</td></tr>';
            try {
                const res = await fetch('/api/alerts?limit=50');
                const data = await res.json();
                if (!data.alerts || data.alerts.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Chưa có dữ liệu cảnh báo nào trong database.</td></tr>';
                    return;
                }
                tbody.innerHTML = data.alerts.map(a => `
                    <tr>
                        <td>#${a.id}</td>
                        <td>${a.timestamp}</td>
                        <td style="color:${a.alert_type === 'fall' ? '#ff5252' : '#ffb74d'}; font-weight:bold;">
                            ${a.alert_type === 'fall' ? '⚠️ TÉ NGÃ' : '🚨 XÂM NHẬP VÙNG'}
                        </td>
                        <td>${a.violated_zones ? 'Vùng ' + a.violated_zones : 'N/A'}</td>
                        <td>
                            <a href="/alerts/${a.image_file}" target="_blank" title="Bấm để xem ảnh lớn">
                                <img src="/alerts/${a.image_file}" class="thumb-img" alt="Snapshot" />
                            </a>
                        </td>
                    </tr>
                `).join('');
            } catch (err) {
                tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#ff5252;">Lỗi tải dữ liệu: ${err.message}</td></tr>`;
            }
        }

        startCamera();
    </script>
</body>
</html>
"""

@app.route('/alerts/<path:filename>')
def get_alert_image(filename):
    return send_from_directory(ALERTS_DIR, filename)

@app.route('/api/alerts', methods=['GET'])
def get_alert_logs():
    limit = request.args.get('limit', default=50, type=int)
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, alert_type, violated_zones, image_file FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in cursor.fetchall()]
        return jsonify({"alerts": rows})
    except Exception as e:
        return jsonify({"error": str(e), "alerts": []}), 500

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
