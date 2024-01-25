from ultralytics import YOLO
import cv2

model = YOLO("/path/to/best.pt")
model.predict(source= "0", show=True, conf=0.7)
