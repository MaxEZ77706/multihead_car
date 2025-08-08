# Main code
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (in one command without extra line breaks)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 "numpy<2" onnxruntime opencv-python-headless ultralytics

# Set working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Run the script
CMD ["python", "main2.py"]
