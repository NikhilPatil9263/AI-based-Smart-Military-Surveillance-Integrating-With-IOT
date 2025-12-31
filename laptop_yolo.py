import cv2
import socket
import numpy as np
import time
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = model = r"C:\Users\nikhi\Downloads\object_detection\runs\military_yolov8m\train_run\weights\best.pt"


RECEIVE_PORT = 6061   # Laptop <-- Pi (frames)
SEND_PORT = 6062      # Laptop --> Pi (detections)

TARGET_FPS = 20       # <<< CHANGE FPS HERE
FRAME_INTERVAL = 1.0 / TARGET_FPS
last_processed_time = 0
# =========================================

model = YOLO(MODEL_PATH)

sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(("", RECEIVE_PORT))
sock_recv.setblocking(False)   # IMPORTANT: drop old frames

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Laptop waiting on port", RECEIVE_PORT, "...")
pi_ip = None

while True:
    # -------- FPS CONTROL --------
    current_time = time.time()
    if current_time - last_processed_time < FRAME_INTERVAL:
        continue
    last_processed_time = current_time
    # ----------------------------

    try:
        packet, addr = sock_recv.recvfrom(65536)
    except BlockingIOError:
        continue

    if pi_ip is None:
        pi_ip = addr[0]
        print("✔ Connected to Raspberry Pi at:", pi_ip)

    data = np.frombuffer(packet, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # -------- YOLO DETECTION --------
    results = model(frame, conf=0.5, verbose=False)
    detections = results[0]

    objects = []
    for box in detections.boxes:
        cls = int(box.cls[0])
        objects.append(model.names[cls])

    result_text = ", ".join(objects) if objects else "No objects"
    sock_send.sendto(result_text.encode(), (pi_ip, SEND_PORT))
    # --------------------------------

    annotated = detections.plot()
    cv2.putText(
        annotated,
        f"YOLO FPS: {TARGET_FPS}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Laptop YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
