FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
# avoid heavy CUDA wheels; use CPU-only stack
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_service.py ./ml_service.py
COPY trainers ./trainers

# writable places for uploads and artifacts
RUN mkdir -p /app/uploads /app/artifacts

EXPOSE 8000
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
