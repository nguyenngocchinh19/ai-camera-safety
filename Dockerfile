FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
VOLUME ["/app/data"]

# Layer 1: Cài PyTorch CPU trước (nặng nhất, ít thay đổi)
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Layer 2: Cài các dependencies còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
