FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent_service.py .
EXPOSE 7000
CMD ["uvicorn", "agent_service:app", "--host", "0.0.0.0", "--port", "7000"]
