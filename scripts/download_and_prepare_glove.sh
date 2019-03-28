#!/usr/bin/env bash

set -e
set -x

EMBEDDINGS_DIR=data/glove

if [ -e ${EMBEDDINGS_DIR}/glove.840B.300d.txt.gz ]
then
    echo "${EMBEDDINGS_DIR}/glove.840B.300d.txt.gzx.txt is found!"
    echo "No need to download the embeddings!"
    exit
fi

wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz

mkdir -p ${EMBEDDINGS_DIR}
mv glove.840B.300d.txt.gz ${EMBEDDINGS_DIR}/glove.840B.300d.txt.gz

