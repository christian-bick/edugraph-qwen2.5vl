#!/bin/bash

# This script should be run on the remote GCP VM, inside the project directory.

# 1. Install AWS CLI and Sync Data from S3
echo "--- Installing AWS CLI ---"
sudo apt-get update && sudo apt-get install -y awscli

echo "--- Cloning Repo ---"
git clone https://github.com/christian-bick/edugraph-qwen2.5vl.git
cd edugraph-qwen2.5vl

echo "--- Syncing data from S3 bucket: s3://imagine-content ---"
# Create the data directory if it doesn't exist
mkdir -p data
# Sync the public bucket. --no-sign-request is for public access.
aws s3 sync s3://imagine-content ./data/ --no-sign-request

# 2. Set up the environment
echo "--- Setting up Python virtual environment ---"
python3 -m venv qwen-env

source qwen-env/bin/activate

echo "--- Installing dependencies from requirements.txt ---"
pip install -r requirements.txt

# 2. Run the training stages in order
echo "--- Starting Stage 1: Knowledge Infusion ---"
python scripts/finetune_stage1_knowledge.py

echo "--- Stage 1 complete. Starting Stage 2: Multimodal Training ---"
python scripts/finetune_stage2_multimodal.py

echo "--- All training stages complete! ---"
