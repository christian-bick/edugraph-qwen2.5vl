#!/bin/bash

# This script creates the VM directly using gcloud compute.

PROJECT_ID=$(grep -o 'project_id = ".*"' gcp/terraform.tfvars | cut -d '"' -f 2)
INSTANCE_NAME="qwen-training-vm"
ZONE="europe-west4-a"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: project_id not found in gcp/terraform.tfvars"
    exit 1
fi

echo "Creating VM instance: $INSTANCE_NAME..."

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --scopes=cloud-platform \
    --metadata=install-gpu-driver=True

echo "VM creation command sent."
