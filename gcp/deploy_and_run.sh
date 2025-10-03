# --- CONFIGURATION: PLEASE EDIT THESE VALUES ---
PROJECT_ID="edugraph-438718"
REGION="europe-west4" # The region for the Artifact Registry
ZONE="europe-west4-a"   # The zone for the Compute Engine VM
# ---

# Define names for the repository and image
REPO_NAME="qwen-25vl-3b"
IMAGE_NAME="qwen-trainer"
IMAGE_TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

# This uses a Container-Optimized OS and runs our container with a GPU attached.
INSTANCE_NAME="qwen-training-vm-docker"
echo "--- Creating VM and starting container... This can take several minutes. ---"

# Debugging: Print all variables to ensure they are set
echo "Instance Name: $INSTANCE_NAME"
echo "Project ID: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Image Tag: $IMAGE_TAG"

gcloud compute instances create-with-container $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --scopes=cloud-platform \
    --container-image=$IMAGE_TAG

echo "\n--- Deployment started! ---"
echo "You can monitor the training by SSHing into the VM and running: docker logs -f $(docker ps -q)"
