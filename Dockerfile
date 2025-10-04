# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (this layer will be cached)
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Dependency Layer ---
# Copy ONLY the requirements file first.
COPY requirements.txt .

# Install dependencies. This layer will only be rebuilt if requirements.txt changes.
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Model Cache Layer ---
# Pre-download the model files. This layer is rebuilt only if the dependencies above change.
RUN python3 -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'; print(f'Downloading files for {model_id}...'); AutoProcessor.from_pretrained(model_id, trust_remote_code=True); Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True); print('Download complete.')"

# --- Google Cloud SDK Layer ---
# Install Google Cloud SDK for gsutil. This is its own layer for better caching.
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/usr/local && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# --- Application Code Layer ---
# Finally, copy the rest of your application code.
# This is the layer that will be rebuilt most often, but it will be very fast.
COPY . .

# Define the default command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]