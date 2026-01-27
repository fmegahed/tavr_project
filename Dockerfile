FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Needed to install older scikit-learn builds (may compile) and common ML deps
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

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Workaround for pycaret 2.3.6 pulling deprecated 'sklearn' package in some cases
# See the error guidance referenced in PyCaret issue threads. :contentReference[oaicite:3]{index=3}
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Pin versions to match the older model ecosystem
# - pycaret==2.3.6 per your note
# - gradio pinned to a 3.x line (3.x works on Python>=3.7). :contentReference[oaicite:4]{index=4}
# - scikit-learn pinned to 0.23.2 which PyCaret 2.x historically targeted. :contentReference[oaicite:5]{index=5}
RUN python -m pip install \
    "pycaret==2.3.6" \
    "scikit-learn==0.23.2" \
    "gradio==3.50.2" \
    -r /app/requirements.txt

COPY . /app

EXPOSE 7860
CMD ["python", "app.py"]
