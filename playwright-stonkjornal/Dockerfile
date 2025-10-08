FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    libglib2.0-0 libnss3 libnspr4 libgtk-3-0 libdrm2 libxkbcommon0 libasound2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    fonts-liberation libwoff1 lsb-release \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

RUN pip install --no-cache-dir -r requirements.txt \
 && python -m playwright install --with-deps chromium

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]

