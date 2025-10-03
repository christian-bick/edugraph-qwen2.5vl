# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir awscli -r requirements.txt

# Pre-download the model and processor to the image cache
RUN python3 -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'; print(f'Downloading files for {model_id}...'); AutoProcessor.from_pretrained(model_id, trust_remote_code=True); Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True); print('Download complete.')"

# Copy only the necessary application code and scripts
COPY . .

# The command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]
