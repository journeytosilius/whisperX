FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install WhisperX and FastAPI
RUN pip install --upgrade pip
RUN pip install git+https://github.com/m-bain/whisperx
RUN pip install fastapi uvicorn python-multipart

# Copy app code
COPY . /app

EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "8000"]
