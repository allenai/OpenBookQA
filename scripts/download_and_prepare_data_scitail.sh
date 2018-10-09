#!/bin/bash

# Fail if any command fails
set -e
set -x

mkdir -p data/
cd data/

set -e

DATASET_URL=http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip

echo "Downloading dataset"
wget $DATASET_URL

echo "Decompressing zip files to `pwd`"
unzip -q SciTailV1.1.zip

# download glove word embeddings
cd ..
bash scripts/download_and_prepare_glove.sh
