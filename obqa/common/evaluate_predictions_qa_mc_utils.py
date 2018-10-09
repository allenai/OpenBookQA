import json
import numpy as np

import torch


def get_single_item_attentions(batch_attentions_item, index, batch_size):
    """
    Gets the values for single item from a batched json item. The batched values such as attentions to facts are
    returned by model execution. However we want to visualize the attentions for a single item from the batch.
    :param batch_attentions_item: Batched json result
    :param index: Index of the item
    :param batch_size: The size of the batch - this is used to determine which fields are actual batched values
    :return:
    """
    if isinstance(batch_attentions_item, dict):
        item = {}
        for k, v in batch_attentions_item.items():
            if isinstance(v, dict):
                # if this is a dict go down the hierarchy
                item[k] = get_single_item_attentions(v, index, batch_size)
            elif isinstance(v, list):
                # If this is a list we check if it is also a batched field (has batch size)
                if len(v) == batch_size:
                    item[k] = v[index]
                elif len(v) > 0:
                    item[k] = [get_single_item_attentions(vv, index, batch_size) for vv in v]
            else:
                item[k] = v
        return item
    else:
        return batch_attentions_item


def dump_data_arc_multi_choice_json(file_handle, metadata, model_output):
    """
    Dumps the prepared exported data to a file
    :param file_handle: The handle of the output file
    :param metadata: Metadata returned by the model
    :param model_output: The output of the model evaluation
    :return: Nothing
    """
    for res in export_output_data_arc_multi_choice_json(metadata, model_output):
        file_handle.write(json.dumps(res))
        file_handle.write("\n")


def export_output_data_arc_multi_choice_json(metadata, model_output):
    """
    Formats the output data to json used for export
    :param file_handle: The handle of the output file
    :param metadata: Metadata returned by the model
    :param model_output: The output of the model evaluation
    :return: Iterator of exported result per item
    """

    batch_size = len(metadata)
    for index, meta in enumerate(metadata):
        res = {}
        res["id"] = meta.get("id", "NA")
        res["question_tokens"] = meta.get("question_tokens", [])
        res["choice_tokens_list"] = meta.get("choice_tokens_list", [])
        res["facts_tokens_list"] = meta.get("facts_tokens_list", [])
        res["gold_label"] = meta.get("label_gold", -1)
        res["gold_facts"] = meta.get("gold_facts", {})

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
                    res["predicted_label"] = int(np.argmax(val))
            elif key == "attentions":
                # visualization happens here
                batch_attentions_item = model_output["attentions"]
                single_attentions_item = get_single_item_attentions(batch_attentions_item, index, batch_size)
                res["attentions"] = single_attentions_item

                if "know_inter_weights" in model_output:
                    res["know_inter_weights"] = model_output["know_inter_weights"]

        yield res


def get_ranks_for_probs(probs):
    """
    Gets the ranks for each of the probabilities.
    :param probs: Probabilities
    :return: Ranks for each of the probability values
    """
    arg2rank = {arg: rank for rank, arg in enumerate(np.argsort(probs)[::-1])}

    return [arg2rank[arg] for arg in range(len(probs))]
