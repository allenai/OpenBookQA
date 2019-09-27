"""
The ``evaluate_predictions_qa_mc`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate_custom --help
    usage: run [command] evaluate_predictions_qa_mc [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data. If no evaluation file is specified it searches for validation_data_file and test_data_file in the model config
    --output_file OUTPUT_FILE
                            path to optional output file with detailed predictions
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any, Iterable
import argparse
from contextlib import ExitStack
import json
import logging

import torch
import tqdm
import numpy as np

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import move_to_device ## for making cuda compatible input tensors

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EvaluatePredictionsQA_MC(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset with optional output'''
        subparser = parser.add_parser('evaluate_predictions_qa_mc',
                                      description=description,
                                      help='Evaluate the specified model + dataset with optional output')
        subparser.add_argument('--archive_file',
                               type=str,
                               required=True,
                               help='path to an archived trained model')
        subparser.add_argument('--evaluation_data_file',
                               type=str,
                               required=False,
                               help='path to the file containing the evaluation data. This can be list of files splitted with ";". If no evaluation file is specified it searches for validation_data_file and test_data_file in the model config')
        subparser.add_argument('--output_file',
                               type=str,
                               required=False,
                               help='output file for raw evaluation results. This can be list of files splitted with ";". At least one value is required that is used as prefix for the output of the results from the evaluation files when evaluation_data_file is not specified! ')
        subparser.add_argument('--cuda_device',
                               type=int,
                               default=-1,
                               help='id of GPU to use (if any)')
        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             output_file: str = None,
             eval_type: str = None) -> Dict[str, Any]:
    model.eval()

    iterator = data_iterator(instances, num_epochs=1)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    with ExitStack() as stack:
        if output_file is None:
            file_handle = None
        else:
            file_handle = stack.enter_context(open(output_file, 'w'))
        for batch in generator_tqdm:
            ## made cuda compatible (if needed)
            batch = move_to_device(batch, cuda_device)
            
            model_output = model(**batch)
            metrics = model.get_metrics()
            if file_handle:
                _persist_data(file_handle, batch.get("metadata"), model_output, eval_type)
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description)

    return model.get_metrics(reset=True)


def _persist_data(file_handle, metadata, model_output, eval_type) -> None:
    if metadata:
        if eval_type == "arc-multi-choice-json" or eval_type == "evaluate_predictions_qa_mc":
            dump_data_arc_multi_choice_json(file_handle, metadata, model_output)
        else:
            _persist_data_default(file_handle, metadata, model_output)


def _persist_data_default(file_handle, metadata, model_output) -> None:
    """
    Outputs the data for default data reader
    :param file_handle: The handle of the output file
    :param metadata: Metadata returned by the model
    :param model_output: The output of the model evaluation
    :return: None, exports data to a given file
    """
    batch_size = len(metadata)
    for index, meta in enumerate(metadata):
        res = {}
        res["id"] = meta.get("id", "NA")
        # We persist model output which matches batch_size in length and is not a Variable
        for key, value in model_output.items():
            if not isinstance(value, torch.autograd.Variable) and len(value) == batch_size:
                val = value[index]
                res[key] = val
        file_handle.write(json.dumps(res))
        file_handle.write("\n")


def dump_data_arc_multi_choice_json(file_handle, metadata, model_output):
    """
    Dumps data for arc_multi_choice_json data reader
    :param file_handle: The handle of the output file
    :param metadata: Metadata returned by the model
    :param model_output: The output of the model evaluation
    :return: None, exports data to a given file
    """

    batch_size = len(metadata)
    for index, meta in enumerate(metadata):
        res = {}
        res["id"] = meta.get("id", "NA")
        res["question_tokens"] = meta.get("question_tokens", [])
        res["choice_tokens_list"] = meta.get("choice_tokens_list", [])
        # We persist model output which matches batch_size in length and is not a Variable
        for key, value in model_output.items():
            if key == "label_probs" or (not isinstance(value, torch.autograd.Variable) and len(value) == batch_size):
                val = value[index]
                if hasattr(val, 'data'):
                    val = val.data.numpy()

                res[key] = val.tolist() if type(val) is np.ndarray else val
                if key == "label_probs":
                    label_ranks = get_ranks_for_probs(val)
                    res["label_ranks"] = label_ranks
                    res["label"] = int(np.argmax(val))
        file_handle.write(json.dumps(res))
        file_handle.write("\n")

def get_ranks_for_probs(probs):
    """
    Gets the ranks for each of the probabilities.
    :param probs: Probabilities
    :return: Ranks for each of the probability values
    """
    arg2rank = {arg: rank for rank, arg in enumerate(np.argsort(probs)[::-1])}

    return [arg2rank[arg] for arg in range(len(probs))]


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader_config = config.pop('dataset_reader')

    if "evaluator_type" in config:
        eval_type = config.get("evaluator_type")
    else:
        dataset_reader_type = dataset_reader_config.get("type")
        eval_type = dataset_reader_type
    dataset_reader = DatasetReader.from_params(dataset_reader_config)


    evaluation_data_paths_list = []
    evaluation_data_short_names = []
    output_files_list = args.output_file.split(";")

    if args.evaluation_data_file:
        evaluation_data_paths_list.append(args.evaluation_data_file)
        evaluation_data_short_names.append("input")
    else:
        if "validation_data_path" in config:
            evaluation_data_paths_list.append(config["validation_data_path"])
            evaluation_data_short_names.append("dev")

        if "test_data_path" in config:
            evaluation_data_paths_list.append(config["test_data_path"])
            evaluation_data_short_names.append("test")

    metrics_out = {}

    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    for i in range(len(evaluation_data_paths_list)):
        evaluation_data_path = evaluation_data_paths_list[i]
        evaluation_data_short_name = evaluation_data_path if len(evaluation_data_short_names) - 1 < i \
                                                          else evaluation_data_short_names[i]

        if len(output_files_list) == len(evaluation_data_paths_list):
            out_file = output_files_list[i]
        else:
            out_file = "{0}_{1}.txt".format(output_files_list[0], evaluation_data_short_name)

        logger.info("Reading evaluation data from %s", evaluation_data_path)
        dataset = dataset_reader.read(evaluation_data_path)

        metrics = evaluate(model, dataset, iterator, args.cuda_device, out_file, eval_type)
        if out_file is not None:
            logging.info("Predictions exported to {0}".format(out_file))

        logger.info("Finished evaluating.")
        logger.info("Metrics:")
        for key, metric in metrics.items():
            logger.info("%s: %s", key, metric)

        if len(evaluation_data_paths_list) == 1:
            metrics_out = metrics
        else:
            metrics_out[evaluation_data_short_name] = metrics

    return metrics_out
