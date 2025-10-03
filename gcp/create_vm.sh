#!/bin/bash

# This script uses gcloud to deploy the Terraform configuration.

# Note: You may need to enable the Infrastructure Manager API the first time you run this:
# gcloud services enable infra-manager.googleapis.com

LOCATION="us-central1"
DEPLOYMENT_NAME="qwen-deployment"

echo "Deploying Terraform configuration using gcloud Infrastructure Manager..."
echo "This may take a few minutes."

# The command must be run from within the 'gcp' directory
cd gcp

gcloud infra-manager deployments apply $DEPLOYMENT_NAME \
    --location=$LOCATION \
    --terraform-blueprint-from-local-path=. \
    --inputs-file=terraform.tfvars

echo "Deployment command sent. Check the GCP console for status."