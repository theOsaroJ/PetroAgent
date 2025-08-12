FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent_service.py ./
EXPOSE 7000
CMD ["uvicorn", "agent_service:app", "--host", "0.0.0.0", "--port", "7000"]
