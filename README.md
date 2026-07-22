<div align="center">

# 🎯 AI-Based Smart Military Surveillance System
### Real-Time Threat Detection & IoT Response

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-FF6B35?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Edge%20Deployment-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.org)
[![License](https://img.shields.io/badge/License-Educational-blue.svg)](#license)

</div>

---

## 📋 Overview

An intelligent surveillance system that detects military threats (soldiers, tanks, weapons, vehicles) in real-time using **YOLOv8 deep learning** and triggers automated responses via **IoT components**. The system combines edge computing (Raspberry Pi) with cloud-grade AI inference, enabling autonomous threat response without human intervention.

**Key Achievement:** 90% mAP50 accuracy, 65+ FPS on Raspberry Pi with ByteTrack object persistence.

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 🤖 **Real-Time Detection** | YOLOv8 with 90% mAP50 accuracy, 151 FPS on Raspberry Pi |
| 🎯 **Military Object Recognition** | Soldiers, tanks, military trucks, guns, aircraft |
| 🚨 **Zone-Based Alerts** | Intrusion detection with configurable polygonal zones |
| 📍 **Object Tracking** | ByteTrack for persistent object identification across frames |
| 📱 **SMS Notifications** | GSM module (SIM800L) sends alerts to authorities in real-time |
| 🚧 **Physical Response** | Servo motor automatically closes barriers on threat detection |
| ⚡ **Edge Optimization** | Runs on Raspberry Pi with minimal latency (no cloud dependency) |
| 🔴 **Local Alerts** | LED and buzzer for immediate on-site notification |
| 🖼️ **Real-Time Visualization** | FPS counter and bounding box annotations with threat highlighting |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              MILITARY SURVEILLANCE SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐                  ┌──────────────────┐ │
│  │  Raspberry Pi    │                  │     Laptop       │ │
│  │  ────────────    │                  │   ──────────    │ │
│  │ • Camera Input   │  UDP Socket      │ • YOLO Inference │ │
│  │ • Frame Capture  │◄───────────────►│ • 90% Accuracy   │ │
│  │ • IoT Control    │   Bi-directional │ • Model Weights  │ │
│  └────────┬─────────┘                  └──────────────────┘ │
│           │                                                   │
│           ├─► THREAT DETECTED ◄─────────────────┐           │
│           │                                       │           │
│      ┌────▼────────────────────────────────────┐ │           │
│      │  Zone Detection & Tracking              │ │           │
│      │  (ByteTrack: Multi-frame persistence)  │ │           │
│      └────┬─────────────────────────────────┬─┘ │           │
│           │                                  │   │           │
│      ┌────▼──────────────┬──────────────────▼┐  │           │
│      │  IoT Responses    │  Alert Outputs    │  │           │
│      ├───────────────────┼──────────────────┤  │           │
│      │ • GSM SMS Alert   │ • LED (Red)       │  │           │
│      │ • Servo Barrier   │ • Buzzer          │  │           │
│      │ • Motion Trigger  │ • Log Recording   │  │           │
│      └───────────────────┴──────────────────┘  │           │
│                                                  │           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Model** | YOLOv8m (Ultralytics) | Military object detection |
| **Tracking** | ByteTrack | Multi-frame object persistence |
| **Computer Vision** | OpenCV | Frame processing & annotation |
| **Deep Learning** | PyTorch + CUDA | GPU-accelerated inference |
| **Edge Hardware** | Raspberry Pi 4/5 | Field deployment |
| **IoT Communication** | GSM Module (SIM800L) | SMS alerts to authorities |
| **Physical Actuators** | Servo Motor | Automated barrier response |
| **Networking** | UDP Sockets | Laptop ↔ Pi communication |
| **Visualization** | OpenCV Display | Real-time FPS monitoring |

---

## 📂 Project Structure

```
AI-based-Smart-Military-Surveillance/
├── yolov8.py                   # Zone detection with ByteTrack (standalone)
├── laptop_yolo.py              # Laptop-side inference (network mode)
├── requirements.txt            # Python dependencies
├── model/
│   └── best.pt                 # Trained YOLOv8m weights (90% mAP50)
├── outputs/                    # Detection results & videos
└── README.md
```

---

## 🎯 Detected Object Classes

```
🪖 Soldier          (Person in military gear)
🚗 Military Truck   (Armed transport vehicles)
🔫 Gun              (Weapons & small arms)
🛡️ Tank             (Armored vehicles)
✈️ Aircraft         (Aerial threats)
```

---

## ⚙️ How It Works

### **Mode 1: Standalone (Single Device)**
```
Webcam/Video File
      ↓
YOLOv8 Detection
      ↓
Zone Check (ByteTrack)
      ↓
Threat Detected?
      ├─ YES → Trigger Alerts (LED, Buzzer, SMS)
      └─ NO → Continue monitoring
```

### **Mode 2: Distributed (Raspberry Pi + Laptop)**
```
Raspberry Pi                          Laptop
┌─────────────────┐                ┌──────────────┐
│ Camera Capture  │                │ YOLO Model   │
│       ↓         │ UDP Stream     │      ↓       │
│  Frame Buffer   │──────────────►│  Detection   │
│       ↑         │◄───────────────│      ↓       │
│ IoT Response    │  Detections    │  Results     │
└─────────────────┘                └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA 11.8+** (for GPU; CPU fallback available)
- **Raspberry Pi 4+** (optional, for edge deployment)
- **GSM Module SIM800L** (optional, for SMS alerts)

### Installation

**1. Clone Repository**
```bash
git clone https://github.com/NikhilPatil9263/AI-based-Smart-Military-Surveillance-Integrating-With-IOT.git
cd AI-based-Smart-Military-Surveillance-Integrating-With-IOT
```

**2. Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**4. Download Model**
Place your trained `best.pt` (YOLOv8m weights) in the `model/` folder:
```
model/
└── best.pt  (≈51 MB)
```

---

## ▶️ Usage

### **Option A: Standalone (Webcam)**
```bash
python yolov8.py
```
- Press `q` to quit
- Configurable parameters in `__main__`:
  - `source_type`: 'webcam', 'video', or 'image'
  - `source_path`: Path to video file
  - `save_output`: Save annotated video

### **Option B: Distributed (Raspberry Pi + Laptop)**

**On Laptop:**
```bash
# Edit laptop_yolo.py:
# Set MODEL_PATH = "path/to/best.pt"
# Set TARGET_FPS = 20  (adjust for your network)
python laptop_yolo.py
# Laptop waits for Pi to send frames
```

**On Raspberry Pi:**
```bash
# Edit laptop_yolo.py with Laptop IP
# Run camera capture script (example):
python rpi_camera_stream.py --laptop-ip 192.168.x.x
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8m |
| **Accuracy (mAP50)** | 90% |
| **FPS (Raspberry Pi)** | 151 |
| **FPS (Laptop i7)** | 60-80 |
| **Latency (Pi→Laptop→Pi)** | ~50ms |
| **Memory (Raspberry Pi)** | ~600 MB |
| **Power Draw** | ~5-8W (Pi only) |
| **Confidence Threshold** | 0.5 (tunable) |

---

## 🔌 IoT Integration

### GSM Module Setup (SIM800L)
```python
# Add to your Pi script:
import serial

gsm = serial.Serial('/dev/ttyUSB0', 9600)
message = "ALERT: Military threat detected at coordinates X,Y"
gsm.write(f'AT+CMGS="+91XXXXXXXXXX"\r')
gsm.write(message + '\x1a')
```

### Servo Motor Control
```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
pwm = GPIO.PWM(17, 50)  # 50Hz frequency

pwm.ChangeDutyCycle(5)   # Barrier closes
time.sleep(2)
pwm.ChangeDutyCycle(10)  # Barrier opens
```

---

## 🎓 Training Your Own Model

To train a custom YOLOv8 model on military object detection:

```bash
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8m.pt')

# Train on custom dataset
results = model.train(
    data='dataset.yaml',      # Your COCO format dataset
    epochs=50,
    imgsz=640,
    device=0,                 # GPU index
    patience=20               # Early stopping
)

# Export weights
model.export(format='pt')
```

---

## 🔮 Roadmap

- [ ] **Drone Detection** — Expand to aerial threats
- [ ] **Low-Light Performance** — Night vision / thermal support
- [ ] **Encrypted Communications** — End-to-end encrypted GSM
- [ ] **Web Dashboard** — Real-time monitoring interface
- [ ] **Multi-Camera Support** — Scale to multiple sensor inputs
- [ ] **Edge TPU** — Google Coral accelerator for faster inference
- [ ] **Cloud Logging** — AWS/GCP event streaming
- [ ] **Automated Training** — Continuous model retraining pipeline

---

## ⚠️ Important Notes

- ⚖️ **Legal Disclaimer:** This system is for **educational and research purposes only**. Ensure compliance with local laws and regulations before deployment in any real-world scenario.
- 🔐 **Privacy & Security:** Implement proper data encryption and access controls for production use.
- 🎯 **Model Accuracy:** Regularly validate on real-world footage and update training data as needed.
- 📡 **Network Reliability:** UDP may drop frames; consider TCP for critical applications.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found** | Verify `best.pt` path in script matches actual location |
| **Low FPS** | Reduce input resolution, decrease confidence threshold, or upgrade GPU |
| **UDP timeout** | Check firewall, ensure Laptop IP is correct in Pi script |
| **Memory overflow on Pi** | Enable swap or reduce frame resolution to 480p |
| **CUDA not available** | Install PyTorch with CUDA support or use CPU (slower) |

---

## 📸 Sample Output

- Annotated frames with bounding boxes → `outputs/`
- Alert logs → Console & SMS (if GSM configured)
- Video recordings (if `save_output=True`) → `outputs/*.mp4`

---

## 💡 Use Cases

- 🏭 **Military Checkpoints** — Border security & perimeter monitoring
- 🌳 **Critical Infrastructure** — Power plants, data centers, airports
- 🚔 **Law Enforcement** — Real-time threat response
- 📹 **Research** — Object detection benchmarking
- 🎓 **Education** — Learning deep learning + IoT integration

---

## 🤝 Contributing

Found a bug? Have ideas for improvements?
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — Object detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
- [OpenCV Tutorials](https://docs.opencv.org/) — Computer vision operations
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/) — Edge deployment

---

## 📄 License

This project is released under the **MIT License** for educational purposes only. See [LICENSE](LICENSE) file for details.

---

## 👨‍💼 Author

**Nikhil Manoj Patil**  
AI/ML Engineer specializing in **Computer Vision**, **Edge Computing**, and **Autonomous Systems**

- 📧 **Email:** nikhilpatil9263@gmail.com
- 💼 **LinkedIn:** [Nikhil Patil](https://linkedin.com/in/nikhil-patil-2013a0282)
- 🐙 **GitHub:** [@NikhilPatil9263](https://github.com/NikhilPatil9263)

---

<div align="center">

⭐ If this project helped you, please consider giving it a star! Your support motivates continued development.

Built with 💙 | YOLOv8 • ByteTrack • PyTorch • OpenCV • Raspberry Pi

</div>
