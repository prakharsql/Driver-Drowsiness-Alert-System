# 🚗 Driver Drowsiness Alert System

A real-time **Driver Drowsiness Detection System** built using **Python, OpenCV, and MediaPipe**.  
The system monitors the driver’s eye state using facial landmarks and triggers a **continuous alert sound** when prolonged eye closure is detected.

---

## 📌 Features
- Real-time webcam-based monitoring
- Accurate eye detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Continuous alert sound for prolonged eye closure
- Face bounding box visualization
- Lightweight and fast execution

---

## 🧠 How It Works
1. Captures live video from the webcam
2. Detects facial landmarks using MediaPipe Face Mesh
3. Extracts eye landmarks and computes **Eye Aspect Ratio (EAR)**
4. If EAR remains below a threshold for a fixed time:
   - 🚨 Triggers a continuous alert sound
5. Alert stops immediately when eyes open

---

## 🛠️ Technologies Used
- Python 3.10
- OpenCV
- MediaPipe
- NumPy
- Winsound (Windows alert)

---

## 📂 Project Structure
