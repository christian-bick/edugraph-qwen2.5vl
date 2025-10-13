# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies in one layer for better caching
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Dependency Layer ---
# Install Google Cloud SDK for gsutil.
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/usr/local
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Copy the requirements file.
COPY requirements.txt .

# Install dependencies using uv.
# We install and use uv in the same layer, calling it via its absolute path
# to avoid any PATH issues within the RUN command's shell.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system --no-cache -r requirements.txt

# --- Model Cache Layer ---
# Define build-time argument for model size
ARG MODEL_SIZE=3b
# Set it as an environment variable to be used in the RUN command
ENV MODEL_SIZE=$MODEL_SIZE
# Pre-download the model files. This layer is rebuilt only if the dependencies above change.
# RUN python3 -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; import os; model_size = os.environ.get('MODEL_SIZE', '3b').upper(); model_id = f'Qwen/Qwen2.5-VL-{model_size}-Instruct'; print(f'Downloading files for {model_id}...'); AutoProcessor.from_pretrained(model_id, trust_remote_code=True); Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True); print('Download complete.')"



# --- Application Code Layer ---
# Finally, copy the rest of your application code.
# This is the layer that will be rebuilt most often, but it will be very fast.
COPY . .

# Define the default command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]
