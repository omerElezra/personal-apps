FROM python:3.11-slim

# Install system dependencies required by Playwright chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-color-emoji \
    xvfb \
    xauth \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# Install Python dependencies and Playwright chromium (without system deps since we installed them manually)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && playwright install chromium \
 && playwright install chrome

ENV PORT=8080
EXPOSE 8080

# IMPORTANT: run your server under xvfb-run (virtual display 1920x1080x24)
CMD ["bash","-lc","xvfb-run -a -s '-screen 0 1920x1080x24' uvicorn app:app --host 0.0.0.0 --port $PORT"]

