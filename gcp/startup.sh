#!/bin/bash
# This script runs automatically on the VM's first boot.

# --- CONFIGURATION: PLEASE EDIT THESE VALUES ---
# Your username on the GCP VM (usually the first part of your Google email)
# e.g., if your email is jane.doe@gmail.com, your username is likely "jane_doe"
REMOTE_USER="christian_bick"

# The full URL to your Git repository
REPO_URL="https://github.com/christian-bick/edugraph-qwen2.5vl.git"
# ---

# Navigate to the user's home directory
cd /home/$REMOTE_USER

# Clone the repository
echo "Cloning repository: $REPO_URL"
git clone $REPO_URL

# Navigate into the repo
# The name is assumed to be the last part of the repo URL
REPO_NAME=$(basename $REPO_URL .git)
cd $REPO_NAME

# Execute the main setup script
echo "Starting main setup and training script..."
bash gcp/setup_and_run.sh
