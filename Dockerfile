# Dockerfile — Inference service with Prometheus metrics

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements_monitoring.txt .
RUN pip install --no-cache-dir -r requirements_monitoring.txt

# Copy application
COPY inference.py .

# Create model directory
RUN mkdir -p /app/model

EXPOSE 5001

CMD ["python", "inference.py"]
