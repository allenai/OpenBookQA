#!/usr/bin/env bash

set -x
set -e

EXPERIMENT_NAME=$1

test -e _experiments && (echo "_experiments exists, bailing" && exit 1)
test -e $EXPERIMENT_NAME && (echo "$EXPERIMENT_NAME exists, bailing" && exit 1)

parallel --halt now,fail=1 --line-buffer -j3 'bash scripts/experiments/qa/run_experiment_openbookqa_gpu.sh' ::: \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice_elmo.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim_elmo.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5wordnet.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5wordnet.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5wordnet.json

mv _experiments $EXPERIMENT_NAME

echo "Success! Results are in $EXPERIMENT_NAME"
