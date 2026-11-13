FROM python:3.11-slim-bookworm AS common
WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    LITTER_DEMO_ASSETS=/app/demo-assets \
    LITTER_DEMO_RESULTS=/app/results
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-demo.txt ./
COPY src/ ./src/
COPY configs/ ./configs/
EXPOSE 7860
CMD ["python", "-m", "litter_detector.demo"]

FROM common AS gpu
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
    && pip install --no-cache-dir -r requirements-demo.txt onnxruntime-gpu==1.26.0 \
    && pip check
ENV LITTER_DEMO_DEVICE=0

FROM common AS cpu
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.9.1+cpu torchvision==0.24.1+cpu \
    && pip install --no-cache-dir -r requirements-demo.txt onnxruntime==1.31.1 \
    && pip check
ENV LITTER_DEMO_DEVICE=cpu
