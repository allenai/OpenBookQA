from typing import Dict, Any
from allennlp.common import Params
from obqa.utils.processing_utils import try_set_val_by_hier_key, clean_split

def update_params(params: Params,
                  update_dict:Dict[str, Any],
                  update_if_exists = False):
    """
    Updates AllenNLP Params object. This is used to automatically parameters of components and then pass them to
    from_params method
    :param params: The params object to update
    :param update_dict: The parameters to update. This is a flat dictionary of {"key1->key2":"value"}
    that updates the params hierarchically where key1 is a parent of key2
    :param update_if_exists: If we want to update the parameter only if it exists.
    :return:
    """
    params_dict = params.as_dict()

    if not update_if_exists:
        params_dict.update(update_dict)
    else:
        for k,v in update_dict.items():
            if k in params_dict:
                params_dict[k] = v
            elif "->" in k:
                try_set_val_by_hier_key(params_dict, clean_split(k, "->"), v, False)

    return Params(params_dict)

