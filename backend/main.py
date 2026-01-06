import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from ultralytics import YOLO
import os

# ================= YOLO MODEL LOADING =================
CUSTOM_MODEL_PATH = "models/yolo_driver.pt"

if os.path.exists(CUSTOM_MODEL_PATH) and os.path.getsize(CUSTOM_MODEL_PATH) > 1_000_000:
    print("✅ Using custom YOLO model")
    yolo = YOLO(CUSTOM_MODEL_PATH)
else:
    print("⚠️ Custom YOLO model not found. Using pretrained yolov8n.")
    yolo = YOLO("yolov8n.pt")  # auto-download

# ================= MediaPipe Face Mesh =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= Eye Landmarks =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= Alarm =================
alarm_active = False

def beep_alarm():
    while alarm_active:
        winsound.Beep(1000, 400)
        time.sleep(0.1)

# ================= Thresholds =================
EAR_THRESHOLD = 0.25
CLOSED_TIME_THRESHOLD = 2.0
eye_closed_start = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    alerts = []

    # ================= FACE & EYE DETECTION =================
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

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > CLOSED_TIME_THRESHOLD:
                alerts.append("DROWSY")
                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=beep_alarm, daemon=True).start()
        else:
            eye_closed_start = None
            alarm_active = False

    # ================= YOLO DETECTION =================
    yolo_results = yolo(frame, conf=0.5, verbose=False)

    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo.names[int(box.cls[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if label == "cell phone":
                alerts.append("PHONE")

    # ================= ALERT TEXT =================
    y_offset = 30
    for text in set(alerts):
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y_offset += 30

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        alarm_active = False
        break

cap.release()
cv2.destroyAllWindows()
