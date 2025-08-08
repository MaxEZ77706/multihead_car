import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import multiprocessing
from PIL import Image
from ultralytics import YOLO

# === PERFORMANCE BOOST ===
cv2.setNumThreads(0)
torch.set_num_threads(min(8, multiprocessing.cpu_count()))
torch.backends.cudnn.benchmark = True

# === CLASS LABELS ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === Multi-Head Model Definition ===
class MobileNetV3_MultiHead(torch.nn.Module):
    def __init__(self, num_classes1=5, num_classes2=5):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.flatten(self.pool(self.features(dummy)))
            in_features = out.shape[1]

        self.head1 = torch.nn.Linear(in_features, num_classes1)
        self.head2 = torch.nn.Linear(in_features, num_classes2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head1(x), self.head2(x)


# === DEVICE & MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3_MultiHead()
model.load_state_dict(torch.load("multihead_best.pth", map_location=device))
model.eval().to(device)


if torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("✅ torch.compile enabled")
    except Exception as e:
        print("⚠️ torch.compile failed, using eager mode:", e)
else:
    print("⚠️ Skipping torch.compile — CUDA not available")


# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# === Load YOLOv8 ===
object_detector = YOLO("yolov8n.pt")


# === VIDEO INFERENCE ===
cap = cv2.VideoCapture("my_cars1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 416
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_multihead.mp4", fourcc, fps, (width, height))
target_width = 640
target_height = 416

frame_id = 0
total_elapsed = 0
show_preview = True

start_global = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    frame = cv2.resize(frame, (width, height))
    results = object_detector.predict(frame, conf=0.4, imgsz=320)

    if results and hasattr(results[0], "boxes") and results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
        boxes = results[0].boxes.xyxy.int().tolist()

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            car_crop = frame[y1:y2, x1:x2]

            if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
                continue

            car_rgb = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(car_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type):
                out1, out2 = model(input_tensor)
                brand_pred = int(torch.argmax(out1, dim=1))
                color_pred = int(torch.argmax(out2, dim=1))

            label = f"{brand_classes[brand_pred]}, {color_classes[color_pred]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

    if show_preview:
        preview = cv2.resize(frame, (target_width, target_height))
        cv2.imshow("Result", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user")
            break

    elapsed = time.time() - start_time
    total_elapsed += elapsed
    print(f"✅ Frame {frame_id} in {elapsed:.2f}s | ⚡ {1 / elapsed:.2f} FPS")

cap.release()
out.release()
cv2.destroyAllWindows()

# === FINAL STATS ===
if frame_id > 0:
    avg_time_per_frame = total_elapsed / frame_id
    avg_fps = frame_id / total_elapsed
else:
    avg_time_per_frame = 0
    avg_fps = 0

print("\n🕒 Processing finished.")
print(f"🧮 Total frames: {frame_id}")
print(f"⏱️ Total time: {total_elapsed:.2f} seconds")
print(f"⚡ Avg time per frame: {avg_time_per_frame:.2f} sec")
print(f"🎯 Avg FPS: {avg_fps:.2f}")
print("🎉 Done: output_multihead.mp4 saved.")
