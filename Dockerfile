# Базовый образ
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-библиотек (одной командой без лишних переносов)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 "numpy<2" onnxruntime opencv-python-headless ultralytics

# Создание рабочей директории
WORKDIR /app

# Копируем весь проект
COPY . /app

# Запуск скрипта
CMD ["python", "main2.py"]
