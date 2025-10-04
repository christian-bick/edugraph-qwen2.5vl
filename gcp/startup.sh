#!/bin/bash
# This script runs automatically on the VM's first boot as the root user.
# It installs Docker, then pulls a Docker image and runs it with GPU support.

# --- Install Google Cloud Ops Agent ---
echo "Installing Google Cloud Ops Agent for GPU monitoring..."
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install

# --- Configuration from Metadata Server ---
echo "--- Reading configuration from metadata server ---"
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"
MODEL_SIZE=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/MODEL_SIZE")
RUN_MODE=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/RUN_MODE")
SKIP_KI=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/SKIP_KI")
PROJECT_ID=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/PROJECT_ID")
REGION=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/REGION")
GCS_BUCKET_FOLDER_PREFIX=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_FOLDER_PREFIX")
GCS_BUCKET_NAME=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_NAME")

# Define names for the repository and image
REPO_NAME="${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}"
IMAGE_NAME="qwen-trainer"
IMAGE_TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

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
# Pass all configuration variables to the container as environment variables
docker run --gpus all --rm \
  -e MODEL_SIZE=$MODEL_SIZE \
  -e RUN_MODE=$RUN_MODE \
  -e SKIP_KI=$SKIP_KI \
  -e PROJECT_ID=$PROJECT_ID \
  -e REGION=$REGION \
  -e GCS_BUCKET_FOLDER_PREFIX=$GCS_BUCKET_FOLDER_PREFIX \
  -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
  "$IMAGE_TAG"

# --- Self-destruct ---
echo "Training process finished. Shutting down the VM."
sudo shutdown -h now

echo "--- Startup script finished ---"
