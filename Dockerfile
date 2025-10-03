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

# Copy only the necessary application code and scripts
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY gcp/ ./gcp/
COPY setup_and_run.sh .

# The command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]