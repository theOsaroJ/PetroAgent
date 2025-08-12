FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_service.py ./
RUN mkdir -p /app/outputs
VOLUME ["/app/outputs"]
EXPOSE 8000
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
