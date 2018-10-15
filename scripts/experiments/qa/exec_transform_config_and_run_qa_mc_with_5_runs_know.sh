#!/bin/bash

set -e
set -x

base_config=$1
config_transform=$2
output_run_dir=$3

temp_dir=$output_run_dir
mkdir -p ${temp_dir}
echo "output_run_dir:${output_run_dir}"
echo "base_config:${base_config}"
echo "beaker_transform:${config_transform}"

random_transformed=${temp_dir}/beaker_config_transformed.json

num_splits=5
for ((run_id=1;run_id<=num_splits;run_id++)); do
    PYTHONPATH=. python tools/config_transform_standalone.py -i ${base_config} -po ${config_transform} -o ${random_transformed}
    temp_config=${temp_dir}/run0${run_id}.json
    cp ${random_transformed} ${temp_config}
done

bash scripts/experiments/qa/exec_run_with_5_runs_partial_know.sh ${temp_dir}

