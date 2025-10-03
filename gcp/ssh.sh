#!/bin/bash

INSTANCE_NAME="qwen-training-vm"
ZONE="us-central1-a"

echo "Connecting to $INSTANCE_NAME via SSH..."

gcloud compute ssh "$INSTANCE_NAME" --zone=$ZONE
