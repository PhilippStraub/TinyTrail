FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# System deps (minimal): libgomp for scikit-learn OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/on-device-drift-detection

# Copy only requirements for layer caching
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Default shell
SHELL ["/bin/bash", "-lc"]

