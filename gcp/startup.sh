#!/bin/bash
# This script runs automatically on the VM's first boot.

# --- CONFIGURATION: PLEASE EDIT THIS VALUE ---
# Your username on the GCP VM (usually the first part of your Google email, e.g., "jane_doe")
REMOTE_USER="christian_bick"

# The full HTTPS URL to your public Git repository
REPO_URL="https://github.com/christian-bick/edugraph-qwen2.5vl.git"
# ---

USER_HOME="/home/$REMOTE_USER"
REPO_NAME=$(basename $REPO_URL .git)
PROJECT_DIR="$USER_HOME/$REPO_NAME"

# Create the user's home directory if it doesn't exist and navigate into it
echo "Setting up in user home directory: $USER_HOME"
mkdir -p $USER_HOME
cd $USER_HOME

# Clone the repository
echo "Cloning public repository: $REPO_URL"
export GIT_TERMINAL_PROMPT=0
git clone $REPO_URL

# Give the user ownership of all the files
chown -R $REMOTE_USER:$REMOTE_USER $PROJECT_DIR

# Execute the main setup script AS THE INTENDED USER
echo "Starting main setup and training script as user $REMOTE_USER..."
sudo -u $REMOTE_USER bash -c "
    set -e # Exit on any error
    cd $PROJECT_DIR
    bash gcp/setup_and_run.sh
"

echo "Startup script finished."
