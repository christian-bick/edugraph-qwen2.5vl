#!/bin/bash

INSTANCE_NAME="qwen-training-vm-docker"
ZONE="europe-west4-a"

echo "This will permanently delete the VM instance '$INSTANCE_NAME'."
read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Deleting VM instance: $INSTANCE_NAME..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE
fi
