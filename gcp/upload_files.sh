#!/bin/bash

INSTANCE_NAME="qwen-training-vm"
ZONE="us-central1-a"
REMOTE_DIR="~/edugraph-qwen2.5vl"

echo "Copying current project directory to $INSTANCE_NAME:$REMOTE_DIR..."

# This command assumes you run it from the root of your project directory.
gcloud compute scp --recurse . "$INSTANCE_NAME:$REMOTE_DIR" --zone=$ZONE

echo "Files copied."
