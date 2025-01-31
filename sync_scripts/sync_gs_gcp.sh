#!/bin/bash

# Function to synchronize source code to Google Cloud Storage
sync_source_code() {
    echo "Synchronizing source code to gcp from Google Cloud Storage"
    gcloud config set storage/parallel_composite_upload_enabled True
    gcloud storage rsync -r -x "datasets/*|exp/*|.*__pycache__.*|weights/*|.git/*" gs://xcloud-shared/masterbin/code/eva-perceiver /home/jupyter/code/eva-perceiver
    echo "done with synchronisation"
}

# Start the synchronization in the background
sync_source_code &

# Your script can continue to run or perform other tasks here

# Wait for background jobs to finish before exiting
wait


# gsutil -m cp -x '/home/jupyter/code/mvlatentdiffusion/mvlatentdiffusion-work-dir' -x '*.pyc' -r /home/jupyter/code/mvlatentdiffusion gs://xcloud-shared/shengyuh/src/mvlatentdiffusion 