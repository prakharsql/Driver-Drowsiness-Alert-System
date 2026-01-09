import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
import os
import sys
import winsound
import threading

# ================= FIX IMPORT PATH =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from email_alert import send_email

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Driver Drowsiness Alert System",
    page_icon="🚗",
    layout="centered"
)

# ================= UI CSS =================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main-title {
    font-size: 38px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff4b4b, #ff9068);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(135deg, #ff4b4b, #ff6f61);
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: 600;
    color: white;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(255,75,75,0.6);
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 2px solid rgba(255,255,255,0.15);
    box-shadow: 0 0 40px rgba(0,0,0,0.7);
    animation: fadeIn 0.4s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🚗 Driver Drowsiness Alert System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time AI Driver Monitoring (MediaPipe + YOLO)</div>', unsafe_allow_html=True)

# ================= SESSION STATE =================
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "last_email" not in st.session_state:
    st.session_state.last_email = 0
if "last_beep" not in st.session_state:
    st.session_state.last_beep = 0

# ================= CONTROLS =================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera", use_container_width=True):
        if not st.session_state.run:
            st.session_state.run = True
            st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with col2:
    if st.button("⏹ Stop Camera", use_container_width=True):
        st.session_state.run = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

st.markdown("---")

frame_box = st.empty()

# ================= MODELS =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

yolo = YOLO("yolov8n.pt")

EAR_THRESHOLD = 0.25
EMAIL_COOLDOWN = 60
BEEP_COOLDOWN = 2  # seconds

# ================= BEEP FUNCTION =================
def play_beep():
    winsound.Beep(1200, 600)  # frequency, duration

# ================= CAMERA LOOP =================
prev_time = time.time()

while st.session_state.run:

    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Camera not accessible")
        break

    alerts = []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_eye = np.array([(face[i].x * w, face[i].y * h) for i in LEFT_EYE])
        right_eye = np.array([(face[i].x * w, face[i].y * h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        if ear < EAR_THRESHOLD:
            alerts.append("DROWSY")

    # ================= YOLO PHONE DETECTION =================
    yolo_results = yolo(frame, conf=0.5, verbose=False)
    for r in yolo_results:
        for box in r.boxes:
            label = yolo.names[int(box.cls[0])]
            if label == "cell phone":
                alerts.append("PHONE")

    # ================= EMAIL ALERT =================
    if alerts and time.time() - st.session_state.last_email > EMAIL_COOLDOWN:
        send_email("🚨 ALERT: " + ", ".join(set(alerts)))
        st.session_state.last_email = time.time()

    # ================= BEEP ALERT =================
    if alerts and time.time() - st.session_state.last_beep > BEEP_COOLDOWN:
        threading.Thread(target=play_beep, daemon=True).start()
        st.session_state.last_beep = time.time()

    # ================= VISUAL ALERT =================
    if alerts:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)
        alert_text = " / ".join(set(alerts))
        cv2.rectangle(frame, (10, 10), (360, 60), (255, 75, 75), -1)
        cv2.putText(frame, alert_text, (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ================= FPS =================
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1]-140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ================= DISPLAY =================
    frame_box.image(frame, channels="BGR", use_container_width=True)
    time.sleep(0.03)

# ================= CLEANUP =================
if st.session_state.cap:
    st.session_state.cap.release()
    st.session_state.cap = None
