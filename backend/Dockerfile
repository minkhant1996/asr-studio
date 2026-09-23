FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch keeps the image at a few GB; swap the index URL for a CUDA build to train on a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8767
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8767"]
