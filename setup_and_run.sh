#!/bin/bash
# This script is the main entrypoint inside the Docker container.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# --- GCS Bucket Configuration ---
GCS_BUCKET="gs://imagine-ml/edugraph-qwen-25vl-3b/"

# --- Test GCS Upload Permission ---
echo "--- Testing GCS upload permission ---"
DUMMY_FILE="gcs_permission_test.txt"
echo "This is a test file to check GCS write permissions." > $DUMMY_FILE

# Try to upload the dummy file. The `|| exit 1` will cause the script to exit if gsutil fails.
echo "Uploading test file to $GCS_BUCKET..."
gsutil cp $DUMMY_FILE ${GCS_BUCKET}${DUMMY_FILE} || exit 1

# If upload was successful, remove the dummy file from the bucket and the local filesystem.
echo "GCS permission test successful. Cleaning up test file..."
gsutil rm ${GCS_BUCKET}${DUMMY_FILE}
rm $DUMMY_FILE

echo "--- GCS permission test passed. Proceeding with main script. ---"

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

# --- Upload results to GCS ---
echo "Uploading adapter models to GCS..."
gsutil -m cp -r out/adapters gs://imagine-ml/edugraph-qwen-25vl-3b/
