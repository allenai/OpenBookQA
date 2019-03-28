#!/bin/bash

set -e
set -x

echo "Starting exec_run_with_5_runs_partial_know.sh"

out_run_dir=$1

if [ ! -n "${out_run_dir}" ] ; then
  echo "${out_run_dir} is empty"
  exit
fi

out_base_dir=${out_run_dir}

mkdir -p ${out_base_dir}

num_splits=5

for ((curr_run=1;curr_run<=num_splits;curr_run++)); do
    echo "curr_run=${curr_run}"
    split_out_dir=${out_base_dir}/run0${curr_run}
    config_file=${out_base_dir}/run0${curr_run}.json

    python -u obqa/run.py train ${config_file} -s ${split_out_dir}
    # evaluate without attentions
    python obqa/run.py evaluate_predictions_qa_mc --archive_file ${split_out_dir}/model.tar.gz --output_file ${split_out_dir}/predictions

    # convert evaluation to aristo-eval json
    python tools/predictions_to_aristo_eval_json.py ${split_out_dir}/predictions_dev.txt > ${split_out_dir}/aristo_evaluator_predictions_dev.txt
    python tools/predictions_to_aristo_eval_json.py ${split_out_dir}/predictions_test.txt > ${split_out_dir}/aristo_evaluator_predictions_test.txt

    # try to export also attentions. This will fail for no-knowledge models
    knowledge_model_name="qa_multi_choice_know_reader_v1"
    if grep -q ${knowledge_model_name} "${config_file}"; then
        echo "${knowledge_model_name} is used in the config ${config_file}. Exporting attentions values for dev and test.."
        python obqa/run.py evaluate_predictions_qa_mc_know_visualize --archive_file ${split_out_dir}/model.tar.gz --output_file ${split_out_dir}/predictions_visual
    fi
    echo "curr_run=${curr_run} - Done!"
done

metrics_files="${out_base_dir}/run01/metrics.json;${out_base_dir}/run02/metrics.json;${out_base_dir}/run03/metrics.json;${out_base_dir}/run04/metrics.json;${out_base_dir}/run05/metrics.json"

python tools/merge_metrics_files.py "${metrics_files}" ${out_base_dir}/metrics.json
echo "The combined metrics from ${num_splits} are printed in ${out_base_dir}/metrics.json"

