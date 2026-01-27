# Use latest Python 3.11 patch release from the official image line
FROM python:3.11-slim

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Gradio defaults (HF Spaces expects port 7860)
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps that commonly help with pycaret[full] and its optional deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    git \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install pycaret full + gradio explicitly.
# Your requirements.txt currently lists "scikit-learn" and "pycaret" :contentReference[oaicite:2]{index=2},
# but you asked specifically for pycaret[full].
RUN python -m pip install \
    "pycaret[full]" \
    gradio \
    -r /app/requirements.txt

# Copy the rest of the repo (app.py, csv, pkl, etc.)
COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
