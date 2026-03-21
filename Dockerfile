FROM python:3.11-slim

WORKDIR /app

# Retry apt in case of transient network issues, use a stable mirror
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (avoids pulling CUDA ~4 GB)
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860 "]

