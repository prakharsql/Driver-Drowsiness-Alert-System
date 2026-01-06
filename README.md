# 🚗 Driver Drowsiness & Distraction Alert System

A real-time **Driver Monitoring System** built using **Python, OpenCV, MediaPipe, and YOLO**.  
The system detects **driver drowsiness and distraction behaviors** such as prolonged eye closure, yawning, and mobile phone usage, and triggers **instant alerts** to enhance road safety.

---

## 📌 Features
- 🎥 Real-time webcam-based monitoring
- 👁️ Eye state detection using **MediaPipe Face Mesh**
- 📐 **Eye Aspect Ratio (EAR)** based drowsiness detection
- 🤖 **YOLO-based object detection** for:
  - 🥱 Yawning detection
  - 📱 Mobile phone usage detection
  - 👀 Driver distraction (looking away)
- 🚨 Continuous alert sound on dangerous conditions
- 🟥 Face & object bounding box visualization
- ⚡ Lightweight and real-time performance

---

## 🧠 How It Works
1. Captures live video from the webcam using **OpenCV**
2. Detects facial landmarks via **MediaPipe Face Mesh**
3. Extracts eye landmarks and computes **EAR**
4. Runs **YOLO object detection** on each frame to detect yawning, phone usage, and distraction
5. Applies **temporal post-detection logic** to reduce false alarms
6. Triggers an alert when unsafe behavior persists
7. Stops alert immediately when normal behavior is restored

---

## 🏗️ System Architecture
Webcam
↓
OpenCV (Frame Capture)
↓
MediaPipe → EAR (Eye Closure)
YOLO → Yawn / Phone / Distraction
↓
Post-Detection Logic
↓
Alert System (Sound + Visual Warning)


---

## 🛠️ Technologies Used
- Python 3.10
- OpenCV
- MediaPipe
- YOLO (Ultralytics)
- NumPy
- Winsound (Windows alert system)

---

## 📂 Project Structure
DRIVER-DROWSINESS-ALERT-SYSTEM/
│
├── .streamlit/
│   └── config.toml
│
├── backend/
│   ├── models/
│   │   └── yolov8n.pt
│   │
│   ├── email_alert.py
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── venv/
│
├── .gitignore
├── README.md
└── requirements.txt

## ⚙️ Installation & Setup
## step_1_clone_repository:
    description: Clone the project repository from GitHub
    commands:
      - git clone https://github.com/your-username/driver-drowsiness-alert.git
      - cd driver-drowsiness-alert

 ## step_2_create_virtual_environment:
    description: Create and activate a virtual environment (optional but recommended)
    commands:
      - python -m venv venv
      - venv\Scripts\activate

 ## step_3_install_dependencies:
    description: Install all required Python packages
    commands:
      - pip install -r requirements.txt
