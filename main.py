from ultralytics import YOLO
import time
from collections import Counter
import torch
import cv2

device = torch.device("cpu")
batch = 8
epochs = 50
imgsz = 1280

if(torch.cuda.is_available()):
    batch = 8
    epochs = 100
    imgsz = 1280
    device = torch.device("cuda")
elif(torch.backends.mps.is_available()):
    batch = 8
    epochs = 100
    imgsz = 1280
    device = torch.device("mps")

def move_servo(angle):
    print("[SERVO] Rotate", angle, "degrees")
    
def pump(delay):
    print("[PUMP] Moves stepper forward for", delay, "seconds")
    time.sleep(delay)
    
def take_photo():
    print("[Camera] Taking photo")
    time.sleep(1)
    
model = YOLO("yolov8s.pt")

def train_model():
    results = model.train(
        data = "datasets/Microplastics.v3i.yolov8/data.yaml",
        epochs = epochs,
        imgsz = imgsz,
        batch = batch,
        device = device,
        name = "microplastics_v1"
    )
    
    print(f"mAP50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall:    {results.results_dict['metrics/recall(B)']:.4f}")
    
    annotated = results.plot()          # draws boxes + labels on image
    cv2.imshow("Detections", annotated)
    cv2.waitKey(0)

if __name__ == '__main__':
    # while True:
    #     move_servo(0)
    #     pump(60)
    #     take_photo()
    #     pump(15)
    train_model()