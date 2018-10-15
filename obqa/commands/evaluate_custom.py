"""
The ``evaluate_custom`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate_custom --help
    usage: run [command] evaluate_custom [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
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

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from torch import Tensor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EvaluateCustom(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset with optional output'''
        subparser = parser.add_parser('evaluate_custom',
                                      description=description,
                                      help='Evaluate the specified model + dataset with optional output')
        subparser.add_argument('--archive_file',
                               type=str,
                               required=True,
                               help='path to an archived trained model')
        subparser.add_argument('--evaluation_data_file',
                               type=str,
                               required=True,
                               help='path to the file containing the evaluation data')
        subparser.add_argument('--output_file',
                               type=str,
                               required=False,
                               help='output file for raw evaluation results')
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
             output_file: str = None) -> Dict[str, Any]:
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
            model_output = model(**batch)
            metrics = model.get_metrics()
            if file_handle:
                id2label = model.vocab.get_index_to_token_vocabulary("labels")
                _persist_data(file_handle, batch.get("metadata"), model_output, id2label=id2label)
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description)

    return model.get_metrics()


def _persist_data(file_handle, metadata, model_output, id2label=None) -> None:
    if metadata:
        batch_size = len(metadata)
        for index, meta in enumerate(metadata):
            res = {}
            res["id"] = meta.get("id", "NA")
            res["meta"] = meta
            # We persist model output which matches batch_size in length and is not a Variable
            for key, value in model_output.items():
                curr_value = None
                if isinstance(value, torch.autograd.Variable) or isinstance(value, Tensor):
                    curr_value = value.data.tolist()

                if not isinstance(curr_value, torch.autograd.Variable) \
                        and isinstance(curr_value, list) \
                        and len(curr_value) == batch_size:
                    val = curr_value[index]
                    res[key] = val

            if "label_probs" in res and id2label is not None:
                labels_by_probs = sorted([[id2label[li], lp] for li, lp in enumerate(res["label_probs"])], key=lambda x:x[1], reverse=True)
                res["labels_by_prob"] = labels_by_probs
                res["label_predicted"] = labels_by_probs[0][0]

            file_handle.write(json.dumps(res))
            file_handle.write("\n")


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
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)

    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    metrics = evaluate(model, dataset, iterator, args.output_file)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
