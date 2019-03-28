#!/usr/bin/env bash

set -e
set -x
	
config_file=$1

if [ ! -n "${config_file}" ] ; then
  echo "${config_file} is empty"
  echo "Use:"
  echo "$0 training_config/qa/multi_choice/openbookqa/your_allennlp_config.json"
  exit
fi

DATASET_NAME_SHORT=obqa
EXPERIMENTS_OUTPUT_DIR_BASE=_experiments

# question and choices
base_config=${config_file}
experiment_prefix_base=${DATASET_NAME_SHORT}_$(basename $base_config)_$(date +%y-%m-%d-%H-%M-%S)-r${RANDOM}
config_transform=training_config/transform_random_seed_and_gpu.json
experiment_out_dir=${EXPERIMENTS_OUTPUT_DIR_BASE}/${experiment_prefix_base}

bash scripts/experiments/qa/exec_transform_config_and_run_qa_mc_with_5_runs_know.sh ${base_config} ${config_transform} ${experiment_out_dir}

