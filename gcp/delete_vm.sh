#!/bin/bash

# This script uses gcloud to delete the Terraform deployment.

LOCATION="us-central1"
DEPLOYMENT_NAME="qwen-deployment"

echo "This will permanently delete all resources managed by the '$DEPLOYMENT_NAME' deployment."
read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Deleting deployment: $DEPLOYMENT_NAME..."
    gcloud infra-manager deployments delete $DEPLOYMENT_NAME --location=$LOCATION
fi