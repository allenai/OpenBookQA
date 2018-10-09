#!/usr/bin/env bash

EMBEDDINGS_DIR=data/glove

if [ -e ${EMBEDDINGS_DIR}/glove.840B.300d.txt.gz ]
then
    echo "${EMBEDDINGS_DIR}/glove.840B.300d.txt.gzx.txt is found!"
    echo "No need to download the embeddings!"
    exit
fi

wget -O glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.840B.300d.zip && gzip glove.840B.300d.txt

mkdir -p ${EMBEDDINGS_DIR}
mv glove.840B.300d.txt.gz ${EMBEDDINGS_DIR}/glove.840B.300d.txt.gz

