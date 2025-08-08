import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD YOLO DETECTOR ===
object_detector = YOLO("yolov8n.pt")

# === LOAD CAR TYPE CLASSIFIER ===
brand_model = mobilenet_v3_large(weights=None)
brand_model.classifier[3] = nn.Linear(1280, 5)
brand_model.load_state_dict(torch.load("car_detection_best_mobilenetv3.pth", map_location=device))
brand_model.to(device).eval()

# === LOAD CAR COLOR CLASSIFIER ===
color_model = mobilenet_v3_large(weights=None)
color_model.classifier[3] = nn.Linear(1280, 5)
color_model.load_state_dict(torch.load("color_detection_best_mobilenetv3.pth", map_location=device))
color_model.to(device).eval()

# === LABELS ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === VIDEO SETUP ===
cap = cv2.VideoCapture("my_cars1.mp4")
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Уменьшение размера в 4 раза
width = original_width // 4
height = original_height // 4

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_type_and_color.mp4", fourcc, fps, (width, height))

frame_id = 0
total_elapsed = 0
start_global = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break
    frame_id += 1
    annotated = frame.copy()

    results = object_detector(frame)
    boxes = results[0].boxes

    if boxes is None or boxes.xyxy.shape[0] == 0:
        resized = cv2.resize(annotated, (width, height))
        out.write(resized)
        continue

    xyxy = boxes.xyxy.int().tolist()
    crop_tensors = []
    valid_boxes = []

    for (x1, y1, x2, y2) in xyxy:
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img)
        crop_tensors.append(input_tensor)
        valid_boxes.append((x1, y1, x2, y2))

    if len(crop_tensors) == 0:
        resized = cv2.resize(annotated, (width, height))
        out.write(resized)
        continue

    batch = torch.stack(crop_tensors).to(device)

    with torch.no_grad():
        brand_preds = brand_model(batch).argmax(1).tolist()
        color_preds = color_model(batch).argmax(1).tolist()

    for (x1, y1, x2, y2), b, c in zip(valid_boxes, brand_preds, color_preds):
        label = f"{brand_classes[b]}, {color_classes[c]}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_resized = cv2.resize(annotated, (width, height))
    out.write(annotated_resized)

    elapsed = time.time() - start_time
    total_elapsed += elapsed
    print(f"✅ Frame {frame_id} done | ⚡ {1 / elapsed:.2f} FPS")

    cv2.imshow("Result", annotated_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# === Final Statistics ===
avg_time_per_frame = total_elapsed / frame_id
avg_fps = frame_id / total_elapsed

print("\n🕒 Processing finished.")
print(f"🧮 Total frames: {frame_id}")
print(f"⏱️ Total time: {total_elapsed:.2f} seconds")
print(f"⚡ Avg time per frame: {avg_time_per_frame:.2f} sec")
print(f"🎯 Avg FPS: {avg_fps:.2f}")
print("🎉 Done: output_type_and_color(yolo).mp4 saved.")
