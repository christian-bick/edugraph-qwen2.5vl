#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- CONFIGURATION: PLEASE EDIT THESE VALUES ---
PROJECT_ID="edugraph-438718"
REGION="europe-west4" # The region for the Artifact Registry
ZONE="europe-west4-a"   # The zone for the Compute Engine VM
# ---

# Define names for the repository and image
REPO_NAME="qwen-25vl-3b"
IMAGE_NAME="qwen-trainer"
IMAGE_TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

# 1. Create Artifact Registry repository (if it doesn't exist)
echo "--- Checking for Artifact Registry repository: $REPO_NAME ---"
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION --project=$PROJECT_ID &> /dev/null; then
    echo "Repository not found. Creating..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for Qwen training"
else
    echo "Repository already exists."
fi

# 2. Configure Docker authentication
echo "--- Configuring Docker to authenticate with GCP... ---"
gcloud auth configure-docker $REGION-docker.pkg.dev

# 3. Build and Push the Docker image
echo "--- Building the Docker image: $IMAGE_TAG ---"
docker build -t $IMAGE_TAG .

echo "--- Pushing the Docker image to Artifact Registry... "
docker push $IMAGE_TAG

