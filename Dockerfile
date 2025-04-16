FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.9 explicitly
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install WhisperX and dependencies

##Downgrade ctranslate to fix a cuda problem
RUN pip install git+https://github.com/m-bain/whisperx
RUN pip install fastapi uvicorn python-multipart torchaudio
RUN pip install ctranslate2==4.4.0


# (Optional) Set environment variables for performance tuning
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Copy app code
COPY . /app

# Expose FastAPI port
EXPOSE 8081

# Run the FastAPI app
CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "8081"]
