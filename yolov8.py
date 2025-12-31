import cv2
from streamlit import video
import torch
from ultralytics import YOLO
from time import time
import os
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path, source_type='webcam', source_path=None, save_output=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        self.model = YOLO(model_path)
        self.source_type = source_type
        self.source_path = source_path
        self.save_output = save_output

        # Define polygonal zone (as list of points)
        self.zone_polygon = np.array([[320, 180], [520, 180], [520, 330], [320, 330]])  # Customizable zone

    def is_inside_zone(self, point):
        return cv2.pointPolygonTest(self.zone_polygon, point, False) >= 0

    def predict_and_draw(self, frame):
        results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml")[0]
        boxes = results.boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [-1]*len(boxes)
        class_names = self.model.names

        alert_triggered = False

        # Draw zone
        cv2.polylines(frame, [self.zone_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        for xyxy, conf, cls_id, obj_id in zip(xyxys, confidences, class_ids, ids):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # center point

            if self.is_inside_zone((cx, cy)):
                alert_triggered = True
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # red dot if in zone
            else:
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            label = f"{class_names[int(cls_id)]} {conf:.2f} ID:{int(obj_id)}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show alert status
        if alert_triggered:
            cv2.putText(frame, "ALERT: Object in Zone", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame, results

    def run_source(self, cap):
        writer = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time()
            frame, _ = self.predict_and_draw(frame)
            end = time()
            fps = 1 / (end - start)

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.save_output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_name = f"{self.source_type}_output.mp4"
                    writer = cv2.VideoWriter(out_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            cv2.imshow("YOLOv8 Zone Alert", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def run(self):
        print(f"[INFO] Running with source_type: {self.source_type}")
        if self.source_type == 'webcam':
            cap = cv2.VideoCapture(0)
        elif self.source_type in ['image', 'video']:
            assert os.path.exists(self.source_path), f"{self.source_type.title()} path not found."
            cap = cv2.VideoCapture(self.source_path)
        else:
            raise ValueError("Invalid source_type. Use 'webcam', 'image', or 'video'.")

        self.run_source(cap)


# 🟢 Main Runner
if __name__ == "__main__":
    print("[INFO] Script started")

    model_path = r"C:\Users\nikhi\Downloads\object_detection\runs\military_yolov8m\train_run\weights\best.pt"
    source_type = 'video'
    source_path = r"C:\Users\nikhi\Downloads\object_detection\img3.jpg"
    save_output = True

    detector = YOLOv8Detector(model_path, source_type, source_path, save_output)
    detector.run()

    print("[INFO] Script finished")
