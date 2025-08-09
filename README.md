# 🚗 Multi-Head Car Classification (Brand & Color)

This project implements a **multi-head deep learning model** for classifying cars by **brand** and **color** from images or video streams.  
It uses **YOLOv8** for detection and a **MobileNetV3-based multi-head classifier** for classification.  
The project supports **PyTorch training**, **ONNX inference**, and **Docker deployment**.

---

## 📂 Project Structure

├── MultiLabel/ # Dataset (ImageFolder format: "<Color> <Brand>")
│ ├── train/
│ ├── val/
│ └── test/
├── train.py # Training script (PyTorch)
├── main.py # Real-time inference with YOLO + ONNX
├── requirements.txt # Python dependencies
├── Dockerfile.infer # Dockerfile for inference
├── Dockerfile.train # Dockerfile for training
└── README.md # Project documentation

## 📊 Features
- **Two outputs**:  
  - Brand: `Motorcycle`, `SUV`, `Sedan`, `Truck`, `Van`  
  - Color: `Black`, `Blue`, `Gray`, `Red`, `White`  
- **ONNX** export for fast inference  
- **Batch processing** for speed optimization  
- **Docker-ready** for reproducible environments  
- **TensorBoard** logging for training visualization

- ## 📦 Installation
1. Clone the repository:  
git clone https://github.com/yourusername/multihead-car.git
cd multihead-car-classification

pip install -r requirements.txt


🏋️ Training
To train the model: python train.py

- To train in Docker:
  1.docker build -t car_train -f Dockerfile.train .
  2.docker run --rm -it \
   -v "$PWD/MultiLabel:/app/MultiLabel" \
   -v "$PWD/logs:/app/runs" \
   -v "$PWD/models:/app/models" \
   car_train

🎯 Inference
Run real-time inference with YOLO + ONNX:  python main.py

Or using Docker:
1.docker build -t car_infer -f Dockerfile.infer .
2.docker run --rm -it \
   -v "$PWD/models:/app/models" \
   -v "$PWD/videos:/app/videos" \
   car_infer

📁 Dataset Format
MultiLabel/

├── train/
│   ├── Black Sedan/
│   ├── Red SUV/
│   └── ...
├── val/
│   ├── Black Sedan/
│   ├── Red SUV/
│   └── ...
└── test/
    ├── Black Sedan/
    ├── Red SUV/
    └── ...

📈 Training Logs
View training metrics with TensorBoard:
tensorboard --logdir=MultiLabel/runs

🛠 Requirements
torch
torchvision
numpy
opencv-python-headless
tensorboard
tqdm
Pillow
