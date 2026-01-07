from dotenv import load_dotenv
import os
load_dotenv("../.env")  # root se load karne ke liye

sid = os.getenv("TWILIO_SID")
token = os.getenv("TWILIO_AUTH")
sender = os.getenv("TWILIO_PHONE")
receiver = os.getenv("ALERT_PHONE")


# Debug prints
print("Twilio SID Loaded:", bool(sid))
print("Twilio Token Loaded:", bool(token))
print("Sender Number Loaded:", sender)
print("Receiver Number Loaded:", receiver)

import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from ultralytics import YOLO
from twilio.rest import Client

# Initialize Twilio Client
client = Client(sid, token)

# YOLO model loading (logic untouched)
CUSTOM_MODEL_PATH = "models/yolo_driver.pt"

if os.path.exists(CUSTOM_MODEL_PATH) and os.path.getsize(CUSTOM_MODEL_PATH) > 1_000_000:
    print("Using custom YOLO model")
    yolo = YOLO(CUSTOM_MODEL_PATH)
else:
    print("Using pretrained YOLOv8n")
    yolo = YOLO("yolov8n.pt")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Alarm flag
alarm_active = False

def beep_alarm():
    while alarm_active:
        winsound.Beep(1000, 400)
        time.sleep(0.2)

# Drowsiness SMS flag to avoid spam
sms_sent = False
eye_closed_start = None

# Camera init
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    alerts = []

    # Face mesh processing (logic untouched)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        # Face bounding box
        xs = [int(lm.x * w) for lm in face]
        ys = [int(lm.y * h) for lm in face]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Eye landmarks
        left_eye = np.array([(face[i].x * w, face[i].y * h) for i in LEFT_EYE])
        right_eye = np.array([(face[i].x * w, face[i].y * h) for i in RIGHT_EYE])

        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < 0.25:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > 2:
                alerts.append("DROWSY")

                # Send SMS only once per drowsy event
                if not sms_sent:
                    sms_sent = True
                    client.messages.create(
                        body="⚠ ALERT: You are feeling drowsy while driving! Please take a break.",
                        from_=sender,
                        to=receiver
                    )
                    print("SMS sent to driver")

                # Start alarm thread if not active
                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=beep_alarm, daemon=True).start()
        else:
            eye_closed_start = None
            alarm_active = False
            sms_sent = False  # reset when driver opens eyes

    # YOLO detections (logic untouched)
    yolo_results = yolo(frame, conf=0.5, verbose=False)

    for r in yolo_results:
        for box in r.boxes:
            label = yolo.names[int(box.cls[0])]
            if label.lower() == "cell phone":
                alerts.append("PHONE")

    # Display frame
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        alarm_active = False
        break

cap.release()
cv2.destroyAllWindows()
