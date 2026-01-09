# ================= STANDARD IMPORTS =================
import sys
import os
import time
import threading
import winsound

# ================= ADD PROJECT ROOT FIRST =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ================= PROJECT IMPORTS =================
from email_alert import send_email

# ================= ENV + TWILIO =================
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(os.path.join(ROOT_DIR, ".env"))

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
ALERT_PHONE = os.getenv("ALERT_PHONE")

print("Twilio SID Loaded:", bool(TWILIO_SID))
print("Twilio Token Loaded:", bool(TWILIO_AUTH))
print("Sender Number:", TWILIO_PHONE)
print("Receiver Number:", ALERT_PHONE)

client = Client(TWILIO_SID, TWILIO_AUTH)

# ================= CV / ML =================
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ================= YOLO MODEL =================
CUSTOM_MODEL_PATH = os.path.join("models", "yolo_driver.pt")

if os.path.exists(CUSTOM_MODEL_PATH) and os.path.getsize(CUSTOM_MODEL_PATH) > 1_000_000:
    print("✅ Using custom YOLO model")
    yolo = YOLO(CUSTOM_MODEL_PATH)
else:
    print("⚠️ Using pretrained YOLOv8")
    yolo = YOLO("yolov8n.pt")

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= LANDMARK INDICES =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
NOSE = 1

# ================= CALCULATIONS =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    return np.linalg.norm(mouth[0] - mouth[1]) / np.linalg.norm(mouth[2] - mouth[3])

# ================= ALARM =================
alarm_active = False

def beep_alarm():
    while alarm_active:
        winsound.Beep(1000, 400)
        time.sleep(0.1)

# ================= THRESHOLDS =================
EAR_THRESHOLD = 0.25
EYE_TIME = 2.0
MAR_THRESHOLD = 0.6
YAWN_TIME = 2.0
LOOK_TIME = 3.0
EMAIL_COOLDOWN = 60
SMS_COOLDOWN = 120

# ================= STATE =================
eye_closed_start = None
yawn_start = None
look_start = None
nose_center = None
last_email_time = 0
last_sms_time = 0
sms_sent = False

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    alerts = []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        # -------- FACE BOX --------
        xs = [int(lm.x * w) for lm in face]
        ys = [int(lm.y * h) for lm in face]
        cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)

        # -------- EYE DROWSINESS --------
        left_eye = np.array([(face[i].x * w, face[i].y * h) for i in LEFT_EYE])
        right_eye = np.array([(face[i].x * w, face[i].y * h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > EYE_TIME:
                alerts.append("DROWSY")

                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=beep_alarm, daemon=True).start()

                if not sms_sent and time.time() - last_sms_time > SMS_COOLDOWN:
                    client.messages.create(
                        body="⚠ ALERT: Drowsiness detected. Please take a break.",
                        from_=TWILIO_PHONE,
                        to=ALERT_PHONE
                    )
                    sms_sent = True
                    last_sms_time = time.time()
        else:
            eye_closed_start = None
            alarm_active = False
            sms_sent = False

        # -------- YAWNING --------
        mouth = np.array([(face[i].x * w, face[i].y * h) for i in MOUTH])
        if mouth_aspect_ratio(mouth) > MAR_THRESHOLD:
            if yawn_start is None:
                yawn_start = time.time()
            elif time.time() - yawn_start > YAWN_TIME:
                alerts.append("YAWNING")
        else:
            yawn_start = None

        # -------- LOOKING AWAY --------
        nose_x = face[NOSE].x * w
        if nose_center is None:
            nose_center = nose_x

        if abs(nose_x - nose_center) > 40:
            if look_start is None:
                look_start = time.time()
            elif time.time() - look_start > LOOK_TIME:
                alerts.append("DISTRACTED")
        else:
            look_start = None

    # -------- PHONE DETECTION --------
    detections = yolo(frame, conf=0.5, verbose=False)
    for r in detections:
        for box in r.boxes:
            label = yolo.names[int(box.cls[0])]
            if label == "cell phone":
                alerts.append("PHONE")

    # -------- EMAIL ALERT --------
    if alerts and time.time() - last_email_time > EMAIL_COOLDOWN:
        send_email("🚨 ALERT: " + ", ".join(set(alerts)))
        last_email_time = time.time()

    # -------- DISPLAY --------
    y = 30
    for a in set(alerts):
        cv2.putText(frame, a, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y += 30

    cv2.imshow("Driver Drowsiness Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
