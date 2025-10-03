#!/bin/bash
# This script runs automatically on the VM's first boot as the root user.

# The full URL to your public Git repository
REPO_URL="https://github.com/christian-bick/edugraph-qwen2.5vl.git"

# Navigate to the root home directory
cd /root/

# Clone the repository
echo "Cloning public repository: $REPO_URL"
export GIT_TERMINAL_PROMPT=0
git clone $REPO_URL

# Navigate into the repo
REPO_NAME=$(basename $REPO_URL .git)
cd $REPO_NAME

# Execute the main setup script
echo "Starting main setup and training script as root..."
chmod +x gcp/setup_and_run.sh
bash gcp/setup_and_run.sh

echo "Startup script finished."
