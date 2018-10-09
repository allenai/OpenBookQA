import json
import numbers
import os
import re
import sys
from copy import deepcopy

import numpy as np


def escape_for_beaker(str):
    return re.sub(r'[^A-Za-z_\-0-9]', '-', str)

if __name__ == "__main__":
    input_files = sys.argv[1]
    out_file = sys.argv[2]
    aggregate_mode = "all"

    default = "avg"
    aggregate_funcs = {
        "avg": np.mean,
        "max": np.max,
        "min": np.min,
        "std": np.std
    }

    add_runs_accuracy_on_level_0 = True
    aggregate_modes = []
    if aggregate_mode == "all":
        aggregate_modes = aggregate_funcs.keys()
    elif ";" in aggregate_mode:
        aggregate_modes = [x.strip() for x in aggregate_mode.split(";")]
    else:
        aggregate_modes = [aggregate_mode]

    input_files_list = [x for x in input_files.split(";") if len(x) > 0]

    full_metrics = {"__raw_metrics": {}}
    combined_metrics_temp = {}

    if add_runs_accuracy_on_level_0:
        flat_metrics_for_runs = {}

    for file_path in input_files_list:
        if not os.path.exists(file_path):
            print("file %s does not exists. skip..." % file_path)
            continue

        file_content = ""
        for line in open(file_path, mode="r"):
            file_content += line.strip()

        file_json = json.loads(file_content)

        full_metrics["__raw_metrics"][file_path] = file_json

        for k, v in file_json.items():
            if k in combined_metrics_temp:
                combined_metrics_temp[k].append(v)
            else:
                combined_metrics_temp[k] = [v]

            if add_runs_accuracy_on_level_0:
                curr_file_k = "_flat_{0}_{1}".format(k, file_path)
                flat_metrics_for_runs[curr_file_k] = v


    merges_cnt = len(full_metrics["__raw_metrics"])
    full_metrics["merges_cnt"] = merges_cnt

    # aggregate metrics
    for mk, mv in combined_metrics_temp.items():
        value_to_save = mv
        if isinstance(value_to_save, list):
            numerics = [x for x in value_to_save if isinstance(x, numbers.Number)]

            if len(numerics) > 0:
                for aggr_mode in aggregate_modes:
                    aggr_v = float(aggregate_funcs[aggr_mode](np.asarray(numerics)))
                    if aggr_mode == default:
                        value_to_save = aggr_v

                    full_metrics[mk + "_" + aggr_mode] = aggr_v

        full_metrics[mk] = value_to_save

    if add_runs_accuracy_on_level_0:
        for k,v in flat_metrics_for_runs.items():
            full_metrics[k] = v

    try:
        if len(full_metrics["__raw_metrics"]) > 0:
            order_func = lambda p_and_metr: (p_and_metr[1]["best_validation_accuracy"], p_and_metr[1]["best_epoch"])
            best_run_path, best_run_metrics = sorted([(key_path, val_metrics) for (key_path, val_metrics) in full_metrics["__raw_metrics"].items()], key=order_func)[-1]

            full_metrics["__best_run_metrics"] = deepcopy(best_run_metrics)
            full_metrics["__best_run_path"] = best_run_path

            # add flat
            if add_runs_accuracy_on_level_0:
                for k,v in best_run_metrics.items():
                    full_metrics["best_run_{0}".format(k)] = v

        else:
            print("No __raw_metrics found!")
    except Exception as e:
        print("Error getting the {0}".format(str(e)))

    # escape metrics-field-names to be compatible with beaker
    full_metrics_new = {}
    for k,v in full_metrics.items():
        full_metrics_new[escape_for_beaker(k)] = v
    with open(out_file, mode="w") as fo:
        fo.write(json.dumps(full_metrics_new, indent=4, sort_keys=True))

