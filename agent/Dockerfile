FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent_service.py .
# api_key.txt is mounted as /run/secrets/openai_api_key by docker-compose
EXPOSE 7000
CMD ["uvicorn", "agent_service:app", "--host", "0.0.0.0", "--port", "7000"]
