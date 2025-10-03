#!/bin/bash

# This script should be run on the remote GCP VM, inside the project directory.

# 1. Set up the environment
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
