#!/bin/bash

# Fail if any command fails
set -e
set -x

mkdir -p data/
cd data/
mkdir -p knowledge

# OpenBookoQA
OPENBOOKQA_DATA_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
wget $OPENBOOKQA_DATA_URL
unzip $(basename $OPENBOOKQA_DATA_URL)

# prepare data for retrieval
cd OpenBookQA-V1-Sep2018
cd Data/Main
# we use the full.jsonl with the retrieval module.
cat train.jsonl dev.jsonl test.jsonl > full.jsonl

# create knowledge file that the data retrieval can read
OPENBOOK_CSV=../../../knowledge/openbook.csv
echo "SCIENCE-FACT" > ${OPENBOOK_CSV}
cat openbook.txt >> ${OPENBOOK_CSV}

cd ../Additional
cat train_complete.jsonl dev_complete.jsonl test_complete.jsonl > full_complete.jsonl

cd ../../../

# knowledge

cd knowledge
CN5_KNOW_DATA_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/knowledge_cn5.zip"
wget $CN5_KNOW_DATA_URL
unzip $(basename $CN5_KNOW_DATA_URL)

# download glove word embeddings
cd ../../
bash scripts/download_and_prepare_glove.sh
