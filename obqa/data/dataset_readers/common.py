"""
Common functions used by the data readers
"""
import json
from typing import Any

from allennlp.common import Params
from allennlp.data import Tokenizer, TokenIndexer


def tokenizer_dict_from_params(params: Params) -> 'Dict[str, Tokenizer]':  # type: ignore
    """
    ``Tokenizer`` can be used in a dictionary, with each ``Tokenizer`` getting a
    name.  The specification for this in a ``Params`` object is typically ``{"name" ->
    {tokenizer_params}}``.  This method reads that whole set of parameters and returns a
    dictionary suitable for use in a ``TextField``.

    Because default values for token indexers are typically handled in the calling class to
    this and are based on checking for ``None``, if there were no parameters specifying any
    tokenizers in the given ``params``, we return ``None`` instead of an empty dictionary.
    """
    tokenizers = {}
    for name, indexer_params in params.items():
        tokenizers[name] = Tokenizer.from_params(indexer_params)
    if tokenizers == {}:
        tokenizers = None
    return tokenizers


def token_indexer_dict_from_params(params: Params) -> 'Dict[str, TokenIndexer]':  # type: ignore
    """
    We typically use ``TokenIndexers`` in a dictionary, with each ``TokenIndexer`` getting a
    name.  The specification for this in a ``Params`` object is typically ``{"name" ->
    {indexer_params}}``.  This method reads that whole set of parameters and returns a
    dictionary suitable for use in a ``TextField``.

    Because default values for token indexers are typically handled in the calling class to
    this and are based on checking for ``None``, if there were no parameters specifying any
    token indexers in the given ``params``, we return ``None`` instead of an empty dictionary.
    """
    token_indexers = {}
    for name, indexer_params in params.items():
        token_indexers[name] = TokenIndexer.from_params(indexer_params)
    if token_indexers == {}:
        token_indexers = None
    return token_indexers


def get_key_and_value_by_key_match(key_value_map, key_to_try, default_key="any"):
    """
    This method looks for a key in a dictionary. If it is not found, an approximate key is selected by checking if the keys match with the end of the wanted key_to_try.
    The method is intended for use where the key is a relative file path!

    :param key_value_map:
    :param key_to_try:
    :param default_key:
    :return:
    """

    retrieved_value = None
    if key_to_try in key_value_map:
        retrieved_value = key_value_map[key_to_try]
        return retrieved_value

    if default_key is not None:
        retrieved_value = key_value_map.get(default_key, None)

    if len(key_value_map) == 1 and default_key in key_value_map:
        retrieved_value = key_value_map[default_key]
    else:
        for key in key_value_map.keys():
            key_clean = key.strip().strip("\"")
            key_clean = key_clean.replace("___dot___", ".")
            if key_clean == "any":
                continue
            if key_to_try.endswith(key_clean):
                retrieved_value = key_value_map[key]
                break

    if retrieved_value is None:
        raise ValueError(
            "key_value_map %s was not matched with a value! Even for the default key %s" % (key_to_try, default_key))

    return retrieved_value


def read_cn5_surface_text_from_json(input_file):
    """
    Reads conceptnet json and returns simple json only with text property that contains clean surfaceText.
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def clean_surface_text(surface_text):
        return surface_text.replace("[[", "").replace("]]", "")

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())
        text = clean_surface_text(item["surfaceText"])
        items.append({"text": text})

    return items


def read_json_flexible(input_file):
    """
    Reads json and returns simple json only with text property that contains clean text.
    Checks several different fields:
    - surfaceText  # conceptnet json
    - tkns
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def clean_surface_text(surface_text):
        return surface_text.replace("[[", "").replace("]]", "")

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())
        if "surfaceText" in item:  # conceptnet
            text = clean_surface_text(item["surfaceText"])
        elif "SCIENCE-FACT" in item:  # 1202HITS
            text = item["SCIENCE-FACT"]
        elif "Row Text" in item:  # WorldTree v1 - it is a typo but we want to handle these as well
            text = item["Row Text"]
        elif "Sentence" in item:  # Aristo Tuple KB v 5
            text = item["Sentence"].replace(".", "").replace("Some ", "").replace("Most ", "").replace("(part)", "")
        elif "fact_text" in item:
            text = item["fact_text"]
        else:
            raise ValueError(
                "Format is unknown. Does not contain of the fields: surfaceText, SCIENCE-FACT, Row Text, Sentence or Sentence!")

        items.append({"text": text})

    return items


def read_cn5_concat_subj_rel_obj_from_json(input_file):
    """
    Reads conceptnet json and returns simple json only with text property that contains clean surfaceText.
    :param input_file: conceptnet json file
    :return: list of items with "text" key.
    """

    def mask_rel(rel):
        return "@{0}@".format(rel.replace("/r/", "cn_"))

    items = []
    for l_id, line in enumerate(open(input_file, mode="r")):
        item = json.loads(line.strip())

        text = " ".join([item["surfaceStart"], mask_rel(item["rel"]), item["surfaceEnd"]])
        items.append({"text": text})

    return items


def load_json_from_file(file_name):
    """
    Loads items from a jsonl  file. Each line is expected to be a valid json.
    :param file_name: Jsonl file with single json object per line
    :return: List of serialized objects
    """
    items = []
    for line in open(file_name, mode="r"):
        item = json.loads(line.strip())
        items.append(item)

    return items


class KnowSourceManager():
    """Class that holds information about knowledge source"""

    def __init__(self,
                 knowledge_source_config_single=None,
                 know_reader=None,
                 know_rank_reader=None,
                 use_know_cache: bool = True,
                 max_facts_per_argument: Any = 0):
        self._knowledge_source_config_single = knowledge_source_config_single  # currently only one source is supported
        self._know_reader = know_reader
        self._know_rank_reader = know_rank_reader
        self._use_know_cache = use_know_cache
        self._max_facts_per_argument = max_facts_per_argument
        self._know_files_cache = {}
        self._know_rank_files_cache = {}

    def get_max_facts_per_argument(self, file_path):
        if isinstance(self._max_facts_per_argument, int):
            return self._max_facts_per_argument
        else:
            max_facts_to_take = get_key_and_value_by_key_match(self._max_facts_per_argument, file_path, "any")
            return max_facts_to_take
