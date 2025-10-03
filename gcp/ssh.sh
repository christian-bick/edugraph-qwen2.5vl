#!/bin/bash

INSTANCE_NAME="qwen-training-vm"
ZONE="europe-west4-a"

echo "Connecting to $INSTANCE_NAME via SSH..."

gcloud compute ssh "$INSTANCE_NAME" --zone=$ZONE
