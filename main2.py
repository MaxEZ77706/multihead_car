import cv2
import torch
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO

# === LOAD YOLO DETECTOR ===
object_detector = YOLO("yolov8n.pt")

# === LOAD ONNX MODELS (без providers) ===
brand_sess = ort.InferenceSession("brand_model.onnx")
color_sess = ort.InferenceSession("color_model.onnx")

# === LABELS ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === NORMALIZATION VALUES ===
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

# === VIDEO SETUP ===
input_path = "my_cars1.mp4"
cap = cv2.VideoCapture(input_path)

target_width = 640
target_height = 416
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_fast_416.mp4', fourcc, fps, (target_width, target_height))

frame_id = 0
total_elapsed = 0
start_global = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break
    
    frame_id += 1
    frame = cv2.resize(frame, (target_width, target_height))
    results = object_detector(frame, conf=0.6, imgsz=416)
    boxes = results[0].boxes.xyxy.int().tolist()

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
            continue

        car_resized = cv2.resize(car_crop, (224, 224))
        car_rgb = cv2.cvtColor(car_resized, cv2.COLOR_BGR2RGB)
        img = car_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        img = ((img - mean) / std).astype(np.float32)

        brand_out = brand_sess.run(None, {"input": img})[0]
        color_out = color_sess.run(None, {"input": img})[0]
        brand_pred = int(np.argmax(brand_out))
        color_pred = int(np.argmax(color_out))

        label = f"{brand_classes[brand_pred]}, {color_classes[color_pred]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

    preview = cv2.resize(frame, None, fx=0.4, fy=0.4)
    cv2.imshow("Result", preview)

    elapsed = time.time() - start_time
    total_elapsed += elapsed
    print(f"✅ Frame {frame_id} processed in {elapsed:.2f} sec | ⚡ {1 / elapsed:.2f} FPS")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("🎉 Done: output_fast_416.mp4 saved.")

avg_time_per_frame = total_elapsed / frame_id
avg_fps = frame_id / total_elapsed

print("\n🕒 Processing finished.")
print(f"🧮 Total frames: {frame_id}")
print(f"⏱️ Total time: {total_elapsed:.2f} seconds")
print(f"⚡ Avg time per frame: {avg_time_per_frame:.2f} sec")
print(f"🎯 Avg FPS: {avg_fps:.2f}")
print("🎉 Done: output_type_and_color.mp4 saved.")
