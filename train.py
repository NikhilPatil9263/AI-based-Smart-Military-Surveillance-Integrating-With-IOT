from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.train(
    data='data.yaml',  # Now with corrected paths
    epochs=100,
    batch=8,
    imgsz=640,
    patience=20,
    augment=True,
    name='military_100epochs_bs8'
)