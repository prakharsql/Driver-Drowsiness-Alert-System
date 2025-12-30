import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Driver Drowsiness Alert System",
    page_icon="🚗",
    layout="wide"
)

# ================= HEADER =================
st.markdown("## 🚗 Driver Drowsiness Alert System")
st.markdown("Real-time eye-based drowsiness detection using **MediaPipe & OpenCV**")
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")
start = st.sidebar.toggle("▶ Start Detection")

EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
CLOSED_TIME_THRESHOLD = st.sidebar.slider("Alert Time (seconds)", 1.0, 5.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Project by Prakhar Deshmukh**")

# ================= METRICS =================
col1, col2, col3 = st.columns(3)
eye_box = col1.metric("👁 Eye Status", "Idle")
time_box = col2.metric("⏱ Closed Time", "0.0 s")
alert_box = col3.metric("🚨 Alert", "OFF")

frame_window = st.image([])

# ================= MEDIAPIPE SETUP =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= ALARM =================
alarm_active = False

def beep_alarm():
    while alarm_active:
        winsound.Beep(1000, 400)
        time.sleep(0.1)

eye_closed_start = None

# ================= MAIN LOOP =================
if start:
    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "No Face"
        closed_time = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
            right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()

                closed_time = time.time() - eye_closed_start
                status = f"DROWSY ({closed_time:.1f}s)"

                eye_box.metric("👁 Eye Status", "Closed")
                time_box.metric("⏱ Closed Time", f"{closed_time:.1f} s")

                if closed_time >= CLOSED_TIME_THRESHOLD:
                    alert_box.metric("🚨 Alert", "ON")
                    if not alarm_active:
                        alarm_active = True
                        threading.Thread(target=beep_alarm, daemon=True).start()
                else:
                    alert_box.metric("🚨 Alert", "OFF")

            else:
                eye_closed_start = None
                alarm_active = False
                status = "Eyes Open"

                eye_box.metric("👁 Eye Status", "Open")
                time_box.metric("⏱ Closed Time", "0.0 s")
                alert_box.metric("🚨 Alert", "OFF")

            color = (0, 255, 0) if status == "Eyes Open" else (0, 0, 255)
            cv2.putText(frame, status, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_window.image(frame, channels="BGR")

    cap.release()
