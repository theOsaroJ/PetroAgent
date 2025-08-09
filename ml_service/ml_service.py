FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force CPU torch wheels (avoid huge CUDA deps)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_service.py .
COPY trainers ./trainers

ENV DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
