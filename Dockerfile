# Use an official NVIDIA CUDA base image, which comes with CUDA and cuDNN pre-installed.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to ensure apt-get runs non-interactively
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python, pip, git, and the AWS CLI
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the working directory
COPY . .

# The command that will be run when the container starts
# It points to our existing setup script.
CMD ["bash", "setup_and_run.sh"]
