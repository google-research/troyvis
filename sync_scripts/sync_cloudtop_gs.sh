#!/bin/bash

# Function to synchronize source code to Google Cloud Storage
sync_source_code() {
    echo "Synchronizing source code to Google Cloud Storage from cloudtop"
    gcloud config set storage/parallel_composite_upload_enabled True
    gcloud storage rsync -r -x "datasets/*|exp/*|.*__pycache__.*|weights/*|.git/*" /usr/local/google/home/masterbin/code/eva-perceiver gs://xcloud-shared/masterbin/code/eva-perceiver
    echo "Done with synchronization"
}

# Start the synchronization in the background
sync_source_code &

# Your script can continue to run or perform other tasks here

# Wait for background jobs to finish before exiting
wait