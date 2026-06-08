# AI Based Military Object Detection Integrating With IoT

A real-time intelligent military surveillance system that detects 
military objects using YOLOv8 and sends alerts via IoT components.

## How It Works
1. Raspberry Pi captures live video and sends frames to Laptop
2. Laptop runs YOLOv8 model and detects military objects
3. Detected objects are sent back to Raspberry Pi
4. Raspberry Pi triggers GSM alert and servo motor

## Files
- `laptop_yolo.py` → Run this on Laptop
- `yolov8.py` → Zone detection with ByteTrack
- `models/best.pt` → Trained YOLOv8 model weights

## ⚠️ Change These Paths Before Running

In `laptop_yolo.py`:
MODEL_PATH = r"your/path/to/models/best.pt"

In `yolov8.py`:
model_path = r"your/path/to/models/best.pt"
source_path = r"your/path/to/test_video.mp4"

## Install Requirements
pip install ultralytics opencv-python torch numpy streamlit

## Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack
- Raspberry Pi
- GSM Module SIM800L
- Servo Motor