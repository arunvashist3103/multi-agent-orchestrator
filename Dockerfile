FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1\
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy application source code.
COPY config.py schemas.py prompts.py graph.py main.py ./

#Run as non-root user for container security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
