#!/bin/bash

# Fail if any command fails
set -e
set -x

mkdir -p data/
cd data/

QUESTIONS_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"

# Download the questions
wget $QUESTIONS_URL
unzip $(basename $QUESTIONS_URL)
mv ARC-V1-Feb2018-2 ARC-V1-Feb2018
rm -rf __MACOSX

# download glove word embeddings
cd ..
bash scripts/download_and_prepare_glove.sh
