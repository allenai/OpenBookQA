#!/bin/bash

# Usage:
# eval_experiment_openbookqa_5runs.sh [DIR_WITH_MODELS_FOR_5RUNS] [evaluate_predictions_qa_mc | evaluate_predictions_qa_mc_know_visualize]
# Example usage:
# bash eval_experiment_openbookqa_5runs.sh know_concepntnet5_5runs/ evaluate_predictions_qa_mc_know_visualize
#  evaluate_predictions_qa_mc - export predictions only
#  evaluate_predictions_qa_mc_know_visualize - export attentions for knowledge-enhanced reader
out_run_dir=$1

if [ ! -n "${out_run_dir}" ] ; then
  echo "${out_run_dir} is empty"
  echo "${out_run_dir} should be a directory, result of 5-run experiment"
  exit
fi

eval_cmd="evaluate_predictions_qa_mc"

if [ -n "$2" ] ; then
eval_cmd=$2
fi

out_base_dir=${out_run_dir}

num_splits=5

for ((curr_run=1;curr_run<=num_splits;curr_run++)); do
    echo "curr_run=${curr_run}"
    split_out_dir=${out_base_dir}/run0${curr_run}

    # evaluate without attentions

    python obqa/run.py ${eval_cmd} --archive_file ${split_out_dir}/model.tar.gz --output_file ${split_out_dir}/predictions

    # convert evaluation to aristo-eval json
    python tools/predictions_to_aristo_eval_json.py ${split_out_dir}/predictions_dev.txt > ${split_out_dir}/aristo_evaluator_predictions_dev.txt
    python tools/predictions_to_aristo_eval_json.py ${split_out_dir}/predictions_test.txt > ${split_out_dir}/aristo_evaluator_predictions_test.txt

    # Uncomment this for exporting attentions! This works when the model is "qa_multi_choice_know_reader_v1"
    # python obqa/run.py evaluate_predictions_qa_mc_know_visualize --archive_file ${split_out_dir}/model.tar.gz --output_file ${split_out_dir}/predictions_visual
    echo "curr_run=${curr_run} - Done!"
done

