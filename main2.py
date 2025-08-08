# batch-inference ONNX модели c imgsz WARMUP

import cv2
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO
import multiprocessing
import torch

# === PERFORMANCE BOOST ===
cv2.setNumThreads(0)  # Отключаем многопоточность OpenCV для лучшей производительности
torch.set_num_threads(min(8, multiprocessing.cpu_count()))
torch.backends.cudnn.benchmark = True

# === CLASS LABELS ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === NORMALIZATION VALUES ===
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

# === Load ONNX Multihead Model ===
providers = ["CPUExecutionProvider"]
multihead_sess = ort.InferenceSession("multihead_batching1.onnx", providers=providers)
input_name = multihead_sess.get_inputs()[0].name

# === Warmup ONNX Multihead Model ===
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
_ = multihead_sess.run(None, {input_name: dummy})

# === Load YOLOv8 Detector ===
object_detector = YOLO("yolov8n.pt")

# === VIDEO SETUP ===
cap = cv2.VideoCapture("my_cars1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 416
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_multihead_batch.mp4", fourcc, fps, (width, height))

frame_id = 0
total_start_time = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    frame = cv2.resize(frame, (width, height))

    # === YOLO DETECTION ===
    with torch.inference_mode():
        results = object_detector(frame, conf=0.4, imgsz=224)
    boxes = results[0].boxes.xyxy.int().tolist()

    crops, coords = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        car_crop = frame[y1:y2, x1:x2]

        if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
            continue

        car_resized = cv2.resize(car_crop, (224, 224))
        car_rgb = cv2.cvtColor(car_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (car_rgb.transpose(2, 0, 1) - mean) / std
        crops.append(img)
        coords.append((x1, y1, x2, y2))

    if crops:
        batch = np.stack(crops, axis=0).astype(np.float32)
        brand_out, color_out = multihead_sess.run(None, {input_name: batch})

        for i, (x1, y1, x2, y2) in enumerate(coords):
            brand_pred = int(np.argmax(brand_out[i]))
            color_pred = int(np.argmax(color_out[i]))
            label = f"{brand_classes[brand_pred]}, {color_classes[color_pred]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

    elapsed = time.time() - start_time
    print(f"✅ Frame {frame_id} in {elapsed:.2f}s | ⚡ {1 / elapsed:.2f} FPS")

# === RELEASE ===
cap.release()
out.release()


# === FINAL STATS ===
total_elapsed = time.time() - total_start_time
print(f"\n🧮 Total frames: {frame_id}")
print(f"⏱️ Total time: {total_elapsed:.2f} sec")
print(f"⚡ Avg FPS: {frame_id / total_elapsed:.2f}")
print("🎉 Done: output_multihead_batch.mp4 saved.")
