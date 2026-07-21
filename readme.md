🎯 AI-Based Smart Military Surveillance Integrating With IoT
A real-time intelligent military surveillance system that automatically detects military objects using YOLOv8 deep learning model and triggers instant alerts through IoT components.

🚀 Features

✅ Real-time military object detection using YOLOv8
✅ Detects soldiers, tanks, aircraft, trucks and guns
✅ Zone-based intrusion detection with alert system
✅ Object tracking using ByteTrack
✅ GSM-based SMS alert to authorities
✅ Servo motor simulation for automatic barrier control
✅ Works on Raspberry Pi (edge deployment)
✅ FPS display and bounding box visualization

output: outputs/output image.jpeg


🛠️ Tech Stack
CategoryTechnologyObject DetectionYOLOv8 (Ultralytics)Object TrackingByteTrackComputer VisionOpenCVDeep LearningPyTorchIoT HardwareRaspberry Pi,  SystemGSM Module SIM800LPhysical ResponseServo MotorUIStreamlitLanguagePython

📁 Project Structure
AI-based-Smart-Military-Surveillance/
├── laptop_yolo.py        → Runs YOLOv8 detection on laptop
├── yolov8.py             → Zone detection with ByteTrack
├── model/
│   └── best.pt           → Trained YOLOv8 model weights
├── outputs/              → Detection output results
├── requirements.txt      → Required libraries
└── README.md

⚙️ How It Works
Raspberry Pi Camera
↓
Captures Live Video Frames
↓
Sends Frames to Laptop via UDP Socket
↓
Laptop runs YOLOv8 Detection
↓
Detected Objects sent back to Raspberry Pi
↓
┌────────────────────┐
│  Threat Detected?  │
└────────────────────┘
↓ YES
┌──────────────┐
│  GSM Alert   │ → SMS to Authorities
│  Servo Motor │ → Barrier Closes
│  LED/Buzzer  │ → Local Alert
└──────────────┘

🎯 Detected Object Classes

🪖 Soldier
🚗 Military Truck
🔫 Gun
🛡️ Tank


📸 Output Results
Check the outputs/ folder for detection results.

🔧 Installation
1. Clone the Repository
git clone https://github.com/NikhilPatil9263/AI-based-Smart-Military-Surveillance-Integrating-With-IOT.git
cd AI-based-Smart-Military-Surveillance-Integrating-With-IOT
2. Install Requirements
pip install -r requirements.txt
3. ⚠️ Update Paths Before Running
In laptop_yolo.py:
MODEL_PATH = r"your/path/to/model/best.pt"
In yolov8.py:
model_path = r"your/path/to/model/best.pt"
source_path = r"your/path/to/test_video.mp4"

▶️ How To Run
Step 1 — Run on Laptop first:
python yolov8.py
Step 2 — Run on Raspberry Pi:
python laptop_yolo.py

📊 Results
MetricValueModelYOLOv8mDetection Speed20 FPSConfidence Threshold0.5Tracked ObjectsSoldiers, Tanks, Aircraft, Trucks, Guns

🔮 Future Scope

🔹 Drone detection support
🔹 Hydraulic barriers instead of servo motors
🔹 Encrypted GSM communication
🔹 Centralized command dashboard
🔹 Night vision / low light detection
🔹 Edge optimization for faster inference


📜 License
This project is for educational purposes only
