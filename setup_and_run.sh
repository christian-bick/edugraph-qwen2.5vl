#!/bin/bash
# This script is the main entrypoint inside the Docker container.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# 1. Sync Data from S3
echo "--- Syncing data from S3 bucket: s3://imagine-content ---"
mkdir -p data
aws s3 sync s3://imagine-content ./data/ --no-sign-request

# 2. Build the training dataset from the synced data
echo "--- Building training dataset from synced data ---"
python3 scripts/build_training_data.py

# 3. Run the training stages in order
if [ "$SKIP_KI" != "true" ]; then
    echo "--- Starting Stage 1: Knowledge Infusion ---"
    python3 scripts/finetune_stage1_knowledge.py
    echo "--- Stage 1 complete. ---"
else
    echo "--- Skipping Stage 1 (Knowledge Infusion) as requested. ---"
fi

echo "--- Starting Stage 2: Multimodal Training ---"
python3 scripts/finetune_stage2_multimodal.py

echo "--- All training stages complete! ---"
