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

# 1. Enable APIs (one-time setup)
#echo "--- Enabling required GCP APIs... ---"
#gcloud services enable artifactregistry.googleapis.com
#gcloud services enable containeranalysis.googleapis.com

# 2. Create Artifact Registry repository (if it doesn't exist)
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

# 3. Configure Docker authentication
echo "--- Configuring Docker to authenticate with GCP... ---"
gcloud auth configure-docker $REGION-docker.pkg.dev

# 4. Build and Push the Docker image
echo "--- Building the Docker image: $IMAGE_TAG ---"
docker build -t $IMAGE_TAG .

echo "--- Pushing the Docker image to Artifact Registry... "
docker push $IMAGE_TAG

# 5. Create a VM and run the container
# This uses a Container-Optimized OS and runs our container with a GPU attached.
INSTANCE_NAME="qwen-training-vm-docker"
echo "--- Creating VM and starting container... This can take several minutes. ---"
gcloud compute instances create-with-container $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --scopes=cloud-platform \
    --container-image=$IMAGE_TAG

echo "\n--- Deployment started! ---"
echo "You can monitor the training by SSHing into the VM and running: docker logs -f $(docker ps -q)"
