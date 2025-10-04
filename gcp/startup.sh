#!/bin/bash
# This script runs automatically on the VM's first boot as the root user.
# It installs Docker, then pulls a Docker image and runs it with GPU support.

# Set PYTHONUNBUFFERED to see logs in real-time
export PYTHONUNBUFFERED=1

# --- Configuration ---
# The full tag of the Docker image to run
IMAGE_TAG="europe-west4-docker.pkg.dev/edugraph-438718/qwen-25vl-3b/qwen-trainer:latest"
# The region where the Artifact Registry is located
REGION="europe-west4"

echo "--- Startup script started ---"

# --- Install Docker ---
echo "Installing Docker..."
# Check if Docker is already installed
if ! command -v docker &> /dev/null
then
    # Install Docker using the official script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    # Add the current user to the docker group to run docker without sudo
    # This is not strictly necessary for the root user, but it's good practice
    usermod -aG docker $(whoami)
else
    echo "Docker is already installed."
fi

# --- Authenticate with Artifact Registry ---
echo "Authenticating with Google Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# --- Pull the Docker image ---
echo "Pulling Docker image: $IMAGE_TAG"
docker pull $IMAGE_TAG

# --- Run the Docker container ---
echo "Running Docker container with GPU support..."
# The --gpus all flag is essential to expose the host's GPUs to the container.
docker run --gpus all --rm "$IMAGE_TAG"

echo "--- Startup script finished ---"
