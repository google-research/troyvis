#!/bin/bash

# Function to synchronize source code to Google Cloud Storage
sync_source_code() {
    echo "Synchronizing source code to cloudtop from Google Cloud Storage"
    gcloud config set storage/parallel_composite_upload_enabled True
    gcloud storage rsync -r -x ".git" gs://xcloud-shared/masterbin/code/eva-perceiver /usr/local/google/home/masterbin/code/eva-perceiver
    # remove unused files
    find . -name "*.pyc" -exec rm -f {} \;
    find . -name "*.so" -exec rm -f {} \;
    rm entrypoint.sh
    rm -rf third_party/d2/build; rm -rf third_party/d2/detectron2.egg-info
    cd third_party/cocoapi/PythonAPI
    rm -rf build; rm -rf pycocotools.egg-info
    cd ../../..
    cd third_party/ops
    rm -rf build; rm -rf dist; rm -rf MultiScaleDeformableAttention.egg-info
    cd ../../
    rm projects/EVAP/clip_vit_base_patch32/pytorch_model.bin
    rm -rf projects/EVAP/clip-vit-large-patch14
    # rm exp
    rm -rf exp
    echo "Done with synchronization"
}

# Start the synchronization in the background
sync_source_code &

# Your script can continue to run or perform other tasks here

# Wait for background jobs to finish before exiting
wait