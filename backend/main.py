import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

# ================= MediaPipe Face Mesh Setup =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= Eye Landmark Indices =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================= EAR Calculation =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= Alarm (Continuous Beep) =================
alarm_active = False

def beep_alarm():
    while alarm_active:
        winsound.Beep(1000, 400)
        time.sleep(0.1)

# ================= Drowsiness Parameters =================
EAR_THRESHOLD = 0.25            # Eye closed threshold
CLOSED_TIME_THRESHOLD = 2.0     # Seconds eyes must be closed
eye_closed_start = None

# ================= Camera =================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # -------- Face Bounding Box --------
        xs = [int(lm.x * w) for lm in face_landmarks]
        ys = [int(lm.y * h) for lm in face_landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      (255, 0, 0), 2)

        # -------- Eye Landmarks --------
        left_eye = np.array([(face_landmarks[i].x * w,
                               face_landmarks[i].y * h) for i in LEFT_EYE])
        right_eye = np.array([(face_landmarks[i].x * w,
                                face_landmarks[i].y * h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2.0

        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        # -------- Drowsiness Logic --------
        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()

            closed_time = time.time() - eye_closed_start
            status = f"DROWSY! {closed_time:.1f}s"
            color = (0, 0, 255)

            if closed_time >= CLOSED_TIME_THRESHOLD and not alarm_active:
                alarm_active = True
                threading.Thread(target=beep_alarm, daemon=True).start()
        else:
            eye_closed_start = None
            alarm_active = False
            status = "Eyes Open"
            color = (0, 255, 0)

        cv2.putText(frame, status, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Driver Drowsiness Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        alarm_active = False
        break

cap.release()
cv2.destroyAllWindows()
