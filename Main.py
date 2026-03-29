import streamlit as st
import cv2
import numpy as np
import os
import json
import urllib.request
import threading
import time
from datetime import datetime, timedelta
import urllib.parse

# ==========================================
# --- CONFIGURATION ---
# ==========================================

# Add as many cameras as you want here!
# Format -> "Location Name": "RTSP Link or USB Port"
camera_password = "Apollo@123"
safe_password = urllib.parse.quote(camera_password)
CAMERAS = {
    "Office":"rtsp://it:Apollo%40123@10.40.23.9:554/Streaming/Channels/102",
    "Front Gate": "rtsp://admin:apollo%401234@10.40.23.117:554/Streaming/Channels/102",
    "Lobby": 0  # You can even mix IP cameras and webcams!
}

FACES_DIR = "faces"
DB_FILE = "attendance.json"
MODELS_DIR = "models"
COOLDOWN_MINUTES = 1

# Stricter recognition (45% match required)
COSINE_THRESHOLD = 0.45 

st.set_page_config(page_title="AI Security", layout="wide")

# ==========================================
# --- BACKGROUND SYSTEM ---
# ==========================================
class SecuritySystem:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Dynamically create a frame storage for every camera in the dictionary
        self.frames = {cam_name: None for cam_name in CAMERAS.keys()}
        self.running = True
        
        self.known_feats = []
        self.known_names = []
       
        # 1. Download Models if missing
        self.download_models()
       
        # 2. Load Faces (Creates a temporary AI just to read the photos)
        self.load_known_faces()
       
        # 3. Start a thread for EVERY camera in the dictionary
        self.threads = []
        for cam_name, cam_url in CAMERAS.items():
            t = threading.Thread(target=self.camera_worker, args=(cam_name, cam_url), daemon=True)
            self.threads.append(t)
            t.start()

    def download_models(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        files = {
            "face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        }
        for file_name, url in files.items():
            path = os.path.join(MODELS_DIR, file_name)
            if not os.path.exists(path):
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(url, path)

    def get_new_ai_models(self):
        # Creates fresh AI models. We need a separate set for every camera thread.
        detector = cv2.FaceDetectorYN.create(
            os.path.join(MODELS_DIR, "face_detection_yunet.onnx"), "", (320, 320), 0.8, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(MODELS_DIR, "face_recognition_sface.onnx"), ""
        )
        return detector, recognizer

    def load_known_faces(self):
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR)
            return

        files = [f for f in os.listdir(FACES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[INFO] Found {len(files)} known face images.")
        
        # Get temporary models just to process the images
        temp_detector, temp_recognizer = self.get_new_ai_models()
       
        for filename in files:
            path = os.path.join(FACES_DIR, filename)
            img = cv2.imread(path)
            if img is None: continue
           
            temp_detector.setInputSize((img.shape[1], img.shape[0]))
            faces = temp_detector.detect(img)
           
            if faces[1] is not None:
                align = temp_recognizer.alignCrop(img, faces[1][0])
                feat = temp_recognizer.feature(align)
                
                # Allows naming like Karthik_1.jpg, Karthik_2.jpg
                raw_name = os.path.splitext(filename)[0]
                clean_name = raw_name.split('_')[0] 
                
                self.known_feats.append(feat)
                self.known_names.append(clean_name)
                print(f"[INFO] Loaded: {clean_name}")

    def log_attendance(self, name, location):
        # Lock file access so two cameras don't write to JSON at the exact same millisecond
        with self.lock:
            if not os.path.exists(DB_FILE):
                with open(DB_FILE, 'w') as f: json.dump([], f)
           
            try:
                with open(DB_FILE, 'r') as f: logs = json.load(f)
            except: logs = []

            now = datetime.now()
            
            # Cooldown check
            for entry in reversed(logs):
                if entry['name'] == name:
                    last_time = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
                    if now - last_time < timedelta(minutes=COOLDOWN_MINUTES):
                        return

            # Append with Location
            logs.append({
                "name": name,
                "location": location,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d")
            })
            with open(DB_FILE, 'w') as f: json.dump(logs, f, indent=4)
            print(f"[SUCCESS] {name} found at {location}")

    def camera_worker(self, cam_name, cam_url):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(cam_url)
        
        # Each camera thread gets its own AI brain to prevent crashing
        detector, recognizer = self.get_new_ai_models()
       
        frame_count = 0
        current_detections = [] 
       
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2)
                cap = cv2.VideoCapture(cam_url)
                continue

            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            frame_count += 1
            
            # Run AI only every 3rd frame
            if frame_count % 3 == 0:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                faces = detector.detect(frame)
                
                current_detections = [] 

                if faces[1] is not None:
                    for face in faces[1]:
                        coords = face[:-1].astype(int)
                        if coords[0]<0 or coords[1]<0 or coords[2]<0 or coords[3]<0: continue

                        align = recognizer.alignCrop(frame, face)
                        if align is None: continue
                       
                        feat = recognizer.feature(align)
                        best_name = "Unknown"
                        max_score = 0.0

                        if self.known_feats:
                            for i, k_feat in enumerate(self.known_feats):
                                score = recognizer.match(k_feat, feat, cv2.FaceRecognizerSF_FR_COSINE)
                                if score > max_score:
                                    max_score = score
                                    best_name = self.known_names[i]

                        if max_score < COSINE_THRESHOLD:
                            best_name = "Unknown"
                        else:
                            # Send the Camera Name (Location) to the logger
                            self.log_attendance(best_name, cam_name)

                        color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
                        current_detections.append((coords, best_name, max_score, color))

            # Draw boxes
            for det in current_detections:
                coords, best_name, max_score, color = det
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                cv2.putText(frame, f"{best_name} ({int(max_score*100)}%)", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save frame to dictionary
            with self.lock:
                self.frames[cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ==========================================
# --- START ---
# ==========================================
@st.cache_resource
def get_system():
    return SecuritySystem()

system = get_system()

# ==========================================
# --- UI ---
# ==========================================
st.title("🏢 Multi-Camera AI Security")

tab1, tab2 = st.tabs(["📋 Activity Logs", "🎥 Live Cameras"])

with tab1:
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Logs"): st.rerun()
    with col2:
        show_all = st.checkbox("Show All History (including past days)")
   
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            try: data = json.load(f)
            except: data = []
       
        if show_all:
            display_data = data
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            display_data = [d for d in data if d.get('date') == today]
       
        if not display_data:
            st.info("No records found.")
        else:
            for entry in reversed(display_data):
                # Dynamically formats: "Karthik found at Office at 14:30:00"
                time_only = entry['timestamp'].split(' ')[1]
                st.success(f"👤 **{entry['name']}** found at **{entry.get('location', 'Unknown Location')}** at 🕒 {time_only}")
    else:
        st.info("No database found.")

with tab2:
    st.caption("Live Feeds")
    run = st.toggle("🔴 Active feeds", value=True)
    
    # Dynamically create columns based on dictionary (2 cameras per row)
    cam_names = list(CAMERAS.keys())
    placeholders = {}
    
    # Create the grid layout
    cols = st.columns(2)
    for i, cam_name in enumerate(cam_names):
        with cols[i % 2]:
            st.subheader(f"📷 {cam_name}")
            placeholders[cam_name] = st.empty()
            st.divider()
   
    if run:
        while run:
            with system.lock:
                # Update every camera placeholder
                for cam_name in cam_names:
                    frame = system.frames[cam_name]
                    if frame is not None:
                        placeholders[cam_name].image(frame, channels="RGB")
                    else:
                        placeholders[cam_name].info(f"Connecting to {cam_name}...")
            time.sleep(0.05)