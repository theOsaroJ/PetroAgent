FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_service.py .

# Secrets (mounted at runtime)
# OPENAI_KEY_FILE env var points to /run/secrets/openai_key_file

EXPOSE 7000
CMD ["uvicorn", "agent_service:app", "--host", "0.0.0.0", "--port", "7000"]
