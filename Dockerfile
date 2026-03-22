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

RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# GOOGLE_FACTCHECK_API_KEY is injected at runtime via HuggingFace Secrets
# Never hardcode API keys here

EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
