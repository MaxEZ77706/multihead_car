import cv2
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter  # ✅ Добавили TensorBoard

# === TENSORBOARD SETUP ===
writer = SummaryWriter(log_dir="logs/eval")

# === PERFORMANCE BOOST ===
cv2.setNumThreads(0)
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
multihead_sess = ort.InferenceSession("multihead.onnx", providers=providers)

# === Load YOLOv8 ===
object_detector = YOLO("yolov8n.pt")

# === Constants ===
YOLO_IMGSZ = 256
CLASSIFIER_SIZE = (224, 224)

# === VIDEO SETUP ===
cap = cv2.VideoCapture("my_cars1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 416
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_multihead_faster.mp4", fourcc, fps, (width, height))

frame_id = 0
total_start_time = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    frame = cv2.resize(frame, (width, height))

    # === YOLO INFERENCE ===
    results = object_detector(frame, conf=0.4, imgsz=YOLO_IMGSZ)
    boxes = results[0].boxes
    if boxes is None or boxes.shape[0] == 0:
        out.write(frame)
        continue

    crops = []
    coords = []

    for box in boxes.xyxy.int().tolist():
        x1, y1, x2, y2 = box[:4]
        car_crop = frame[y1:y2, x1:x2]

        if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
            continue

        car_resized = cv2.resize(car_crop, CLASSIFIER_SIZE, interpolation=cv2.INTER_LINEAR)
        car_rgb = cv2.cvtColor(car_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = car_rgb.transpose(2, 0, 1)
        img = (img - mean) / std
        crops.append(img)
        coords.append((x1, y1, x2, y2))

    # === ONNX MULTIHEAD INFERENCE ===
    if crops:
        batch = np.stack(crops, axis=0).astype(np.float32)
        brand_out, color_out = multihead_sess.run(None, {"input": batch})

        for i in range(len(crops)):
            input_tensor = np.expand_dims(crops[i], axis=0).astype(np.float32)
            brand_out, color_out = multihead_sess.run(None, {"input": input_tensor})

            x1, y1, x2, y2 = coords[i]
            brand_pred = int(np.argmax(brand_out))
            color_pred = int(np.argmax(color_out))
            label = f"{brand_classes[brand_pred]}, {color_classes[color_pred]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    preview = cv2.resize(frame, None, fx=0.6, fy=0.6)
    cv2.imshow("Faster Detection", preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start_time
    print(f"✅ Frame {frame_id} in {elapsed:.2f}s | ⚡ {1 / elapsed:.2f} FPS")
    writer.add_scalar("FPS/Per_Frame", 1 / elapsed, frame_id)

cap.release()
out.release()
cv2.destroyAllWindows()

# === Summary ===
if frame_id > 0:
    total_elapsed = time.time() - total_start_time
    avg_fps = frame_id / total_elapsed
    avg_time_per_frame = total_elapsed / frame_id

    print("\n🕒 Processing finished.")
    print(f"🧮 Total frames: {frame_id}")
    print(f"⏱️ Total time: {total_elapsed:.2f} seconds")
    print(f"⚡ Avg time per frame: {avg_time_per_frame:.2f} sec")
    print(f"🎯 Avg FPS: {avg_fps:.2f}")
    print("🎉 Done: output_multihead_faster.mp4 saved.")

    # === WRITE FINAL METRICS TO TENSORBOARD ===
    writer.add_scalar("FPS/Average", avg_fps, 0)
    writer.add_scalar("Time/Total", total_elapsed, 0)
    writer.add_scalar("Frames/Total", frame_id, 0)
    writer.flush()
else:
    print("❌ No frames processed. Check your video path or YOLO results.")

writer.close()
