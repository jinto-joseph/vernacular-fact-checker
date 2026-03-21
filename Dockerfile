FROM python:3.11-slim

WORKDIR /app

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    supervisor \
 && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Supervisor config: run API on 7860, Streamlit on 8501
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
