"""
This scripts applies transformations from one allennlp config file to another.
Python Random transformations are supported.
"""
import argparse
import json
import numbers
import os
import random

from typing import List, Any, Dict

from allennlp.common import Params

#from data.utils.processing_utils import clean_split, get_fields_with_str_values_from_txt, try_set_val_by_hier_key
from typing import List, Any

import _jsonnet

def clean_split(text, delim):
    """
    Split texts and removes unnecessary empty spaces or empty items.
    :param text: Text to split
    :param delim: Delimiter
    :return: List of splited strings
    """

    return [x1 for x1 in [x.strip() for x in text.split(delim)] if len(x1)>0]


def get_fields_from_txt(txt, field_delim=";", hier_delim="->", name_mapping_delim=":"):
    """
    Parses a field setup to dict of ("new_field_name", ["json_hier1", "json_heir2"]) given text
    :param txt: Example setting "new_field_name:json_hier1->json_heir2;new_field_name_2:json_hierA1->json_heirA2;"
    :param field_delim: Delimiter between field mappings. Default is semicolon ;
    :param hier_delim: Hierarchical delimiter between json fields. Default "->"
    :param name_mapping_delim: Delim between new field name and json fields. Default double dot ":"
    :return: Dictionary of (field_name, json_hier_fields_list)
    """

    splitted_fields = clean_split(txt, field_delim)

    named_splitted_fields = [(None, clean_split(x[0]), hier_delim) if len(x) == 1 else (x[0], clean_split(x[0], hier_delim)) for x in [clean_split(x, name_mapping_delim) for x in splitted_fields]]

    dict_name_to_key = {"__".join(k) if n is None else n :k for n,k in named_splitted_fields}

    return dict_name_to_key


def get_val_by_hier_key(json_item,
            hier_key:List[str],
            raise_key_error=False,
            default=None):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item
    res_val = default

    found = True
    for fld in hier_key:
        if not fld in curr_obj:
            found = False
            if raise_key_error:
                raise KeyError("Key {0} not found in object json_item. {1}".format("->".join(["*%s*" % x if x==fld else x  for x in hier_key]),
                                                                                   "Starred item is where hierarchy lookup fails!" if len(hier_key) > 1 else "" ))
            break
        curr_obj = curr_obj[fld]

    if found:
        res_val = curr_obj

    return res_val


def try_set_val_by_hier_key(json_item,
                            hier_key: List[str],
                            value: Any,
                            create_hier_if_not_exists=True):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item

    success = False
    for fid, fld in enumerate(hier_key):
        if isinstance(curr_obj, list):
            is_int = False
            try:
                fld_as_int = int(fld)
                is_int = True
            except ValueError:
                raise

            if is_int:
                if fid == (len(hier_key) - 1):  # if this is the last field in the hierarchy to set
                    curr_obj[fld_as_int] = value
                    success = True
                else:
                    curr_obj = curr_obj[fld_as_int]

        elif isinstance(curr_obj, dict):
            if not fld in curr_obj:
                if create_hier_if_not_exists:
                    curr_obj[fld] = {}
                else:
                    raise KeyError("Key {0} not found in object json_item. {1}".format(
                        "->".join(["*%s*" % x if x == fld else x for x in hier_key]),
                        "Starred *item* is where hierarchy lookup fails!" if len(hier_key) > 1 else ""))

            if fid == (len(hier_key) - 1):  # if this is the last field in the hierarchy to set
                curr_obj[fld] = value
                success = True
            else:
                curr_obj = curr_obj[fld]
        else:
            raise ValueError("Object {0} should be dictionary or list to do lookup.")


    return success


def test_try_set_val_by_hier_key():

    json = {}

    assert try_set_val_by_hier_key(json, ["value1", "value2", "value3"], value=[1, 2, 3], create_hier_if_not_exists=True)
    assert try_set_val_by_hier_key(json, ["value1", "value2", "value3", "2"], value=4)

    assert json["value1"]["value2"]["value3"][2] == 4

    failed = False
    try:
        try_set_val_by_hier_key(json, ["value1", "value2", "value3", "4"], value=4)
    except:
        failed = True
    assert failed


def get_fields_with_str_values_from_txt(txt, field_delim="|", hier_delim="->", value_mapping_delim=":", ):
    """
    Parses a field setup to dict of (["json_hier1", "json_heir2"], value) given text
    :param txt: Example setting "json_hier1->json_heir2=>new_val;json_hierA1->json_heirA2=>new_val;"
    :param field_delim: Delimiter between field mappings. Default is semicolon ;
    :param hier_delim: Hierarchical delimiter between json fields. Default "->"
    :param value_mapping_delim: Delim between new field name and json fields. Default double dot ":"
    :return: Dictionary of (field_name, json_hier_fields_list)
    """

    transformations_list = clean_split(txt, field_delim)

    splitted_str_fields_with_values = [tuple(clean_split(field, value_mapping_delim)) for field in transformations_list]

    field_hier_with_str_values = [(clean_split(fv[0], hier_delim), fv[1]) for fv in splitted_str_fields_with_values]

    return field_hier_with_str_values


def assert_list_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)


def test_get_fields_with_str_values_from_txt():
    # test multiple fields with multiple hierarchy
    transofrmations = "test1->test2->test3=>new_val1| test4->test5=>new_val2| test6=>  new_val3 |"

    expected = [(["test1", "test2", "test3"], "new_val1"),
                (["test4", "test5"], "new_val2"),
                (["test6"], "new_val3"),
                ]
    parsed = get_fields_with_str_values_from_txt(transofrmations, field_delim="|", hier_delim="->",
                                                 value_mapping_delim="=>")

    assert_list_equal(parsed[0][0], expected[0][0])
    assert parsed[0][1] == expected[0][1]

    assert_list_equal(parsed[1][0], expected[1][0])
    assert parsed[1][1] == expected[1][1]

    assert_list_equal(parsed[2][0], expected[2][0])
    assert parsed[2][1] == expected[2][1]


def get_random_value(config: Dict[Any, Any]):
    """
    Executes function from https://docs.python.org/3/library/random.html
    :param config: Config that contains type "get_random_value" with 'func' and 'args'
    :return:
    """
    config_dict = config
    if config_dict["type"] != "random":
        raise ValueError("Config with type {0} is not valid for this function".format(config_dict["type"]))

    func_str = config_dict["func"]
    func_args = dict(config_dict["args"])

    return_first = False
    if func_str.endswith("[0]"):
        return_first = True
        func_str = func_str[:-3]

    allowed_funcs = {
        "random.randint": random.randint,
        "randint": random.randint,
        "random.uniform": random.uniform,
        "uniform": random.uniform,
        "random.randrange": random.randrange,
        "randrange": random.randrange,
        "random.shuffle": random.shuffle,
        "shuffle": random.shuffle,
        "random.choice": random.choice,
        "choice": random.choice,
        "random.choices": random.choices,
        "choices": random.choices,
    }

    if not func_str in allowed_funcs:
        raise ValueError("Wrong function type {0}. The following types are allowed: {1}".format(func_str, ", ".join(sorted(allowed_funcs.keys()))))

    func = allowed_funcs[func_str]

    result_value = func(**func_args)
    if return_first:
        result_value = result_value[0]

    return result_value

def _replace_randoms(dictionary: dict) -> Dict[str, Any]:
    """
    Replaces a random() functions with the executed random value.
    :param dictionary: Json object
    :return: Json object with random values
    """

    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            if "type" in dictionary[key] and dictionary[key]["type"] == "random":
                dictionary[key] = get_random_value(dictionary[key])
            else:
                dictionary[key] = _replace_randoms(dictionary[key])
        elif isinstance(dictionary[key], list):
            for item in dictionary[key]:
                if isinstance(item, dict):
                    _replace_randoms(item)
    return dictionary

def _replace_value(dictionary: Dict[str, Any], value_to_find, value_replace) -> Dict[str, Any]:
    """
    Replaces values by searching in the tree.
    :param dictionary: dict object
    :param value_to_find: The value to find
    :param value_replace: The value to replace
    :return:
    """

    for key in dictionary.keys():
        if dictionary[key] == value_to_find:
            dictionary[key] = value_replace
        elif isinstance(dictionary[key], dict):
            dictionary[key] = _replace_value(dictionary[key], value_to_find, value_replace)
    return dictionary

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Transform config file given a list of json hierachical transformations ')
    parser.add_argument('-i', '--input_file', dest="input_file", metavar='CONFIG_FILE', type=str,
                        help='Input config file to transform. The file is loaded and json settings with type random are evaluated.')

    parser.add_argument('-t', '--transformations', dest="transformations", metavar='level1->level2:5|field2:[1,2,3]', type=str,
                        help='transformations to apply', default=None)

    parser.add_argument('-po', '--params_overrides', dest="params_overrides", type=str,
                        help='Params overrides to apply. This is a json file with updated values. It can also be a path to a json file!.', default=None)

    parser.add_argument('-o', '--output_file', dest="output_file", metavar='NEW_CONFIG_FILE', type=str, default=None,
                        help='File to save the new config')

    args = parser.parse_args()

    config_json = _replace_randoms(Params.from_file(args.input_file).as_dict())

    if args.params_overrides is not None:
        # handle json overrides

        overrides_dict = None
        if os.path.exists(args.params_overrides):
            with open(args.params_overrides) as po_f:
                # overrides_dict = pyhocon.ConfigFactory.parse_string(po_f.read())
                input_str = po_f.read()
                overrides_dict = json_parse = json.loads(_jsonnet.evaluate_snippet("snippet", input_str))
        else:
            input_str = args.params_overrides
            overrides_dict = json_parse = json.loads(_jsonnet.evaluate_snippet("snippet", input_str))

        overrides_dict = _replace_randoms(overrides_dict)

        new_overrides_dict = {}
        # handle also dict of hierarchical keys
        for k, v in overrides_dict.items():
            if "->" in k:
                hier_keys = clean_split(k, "->")
                try_set_val_by_hier_key(new_overrides_dict, hier_keys, v, create_hier_if_not_exists=True)
            else:
                new_overrides_dict[k] = v

        config_json = Params.from_file(args.input_file, params_overrides=json.dumps(new_overrides_dict)).as_dict()

    if args.transformations is not None:
        # handle transformation in the format 'level1->level2:5|field2:[1,2,3]'
        parsed_transformations_list = get_fields_with_str_values_from_txt(args.transformations)

        # update values
        for tr in parsed_transformations_list:
            hier_keys = tr[0]
            value = tr[1]
            try_set_val_by_hier_key(config_json, hier_keys, value, create_hier_if_not_exists=False)

    config_json = _replace_value(config_json, None, "None")
    if args.output_file:
        with open(args.output_file, mode="w") as fout:
            json.dump(config_json, fout, indent=4, sort_keys=True)
    else:
        print(json.dumps(config_json, indent=4, sort_keys=True))






