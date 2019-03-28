#!/usr/bin/env bash

set -e
set -x

# example for running the comparison
export data=test


python tools/aristo_eval_json_compare_multiple_outputs.py \
--gold \
data/OpenBookQA-V1-Sep2018/Data/Main/${data}.jsonl \
--files \
model1_output/aristo_evaluator_predictions_${data}.txt \
model2_output/aristo_evaluator_predictions_${data}.txt \
model3_output/aristo_evaluator_predictions_${data}.txt \
--friendly_names \
model1 \
model2 \
model3 \
--out \
openbookqa_comparison_models123_${data}.csv
