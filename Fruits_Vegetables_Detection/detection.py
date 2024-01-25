from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/user/Desktop/Food_detection2/Meyve_Sebze.pt")
model.predict(source= "0", show=True, conf=0.7)