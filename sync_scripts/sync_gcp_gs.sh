#!/bin/bash

# Function to synchronize source code to Google Cloud Storage
sync_source_code() {
    echo "Synchronizing source code to Google Cloud Storage from gcp"
    gcloud config set storage/parallel_composite_upload_enabled True
    gsutil -m rsync -r -d -x "datasets/*|exp/*|.*__pycache__.*|weights/*|.git/*" /home/jupyter/code/eva-perceiver gs://xcloud-shared/masterbin/code/eva-perceiver
}

# Start the synchronization in the background
sync_source_code &

# Your script can continue to run or perform other tasks here

# Wait for background jobs to finish before exiting
wait