#!/bin/bash

# Fail if any command fails
set -e
set -x

TRAINED_MODELS=data/trained_models
mkdir -p ${TRAINED_MODELS}

MODELS_URL_BASE=https://s3-us-west-2.amazonaws.com/ai2-website/data/aristo-obqa
model_files_list=(
model_q2ch_best_run.tar.gz
model_esim_best_run.tar.gz
model_kn_conceptnet5_and_openbook_best_run.tar.gz
model_kn_conceptnet5_best_run.tar.gz
model_kn_wordnet_and_openbook_best_run.tar.gz
model_kn_wordnet_best_run.tar.gz
model_esim_elmo_best_run.tar.gz
model_q2ch_elmo_best_run.tar.gz
)

runs_num=8
for ((file_id_curr=0;file_id_curr<${runs_num};file_id_curr++)); do
    model_file_curr=${model_files_list[${file_id_curr}]}
    wget -O ${TRAINED_MODELS}/${model_file_curr} $MODELS_URL_BASE/${model_file_curr}
done

