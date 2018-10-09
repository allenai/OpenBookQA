import _jsonnet
import json


def jsonnet_loads(jsonnet_str, ext_vars=None):
    """
    Parses jsonnet string into json
    :param jsonnet_str: Jsonnet function
    :param ext_vars: External vars that can be passed as {'SOME_PARAM': 'AI2'} and used in the jsonnet as {name: std.extVar("SOME_PARAM")}
    :return:
    """
    json_parse = json.loads(_jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars))

    return json_parse


def jsonnet_load_file(file_name):
    """
    Parses jsonnet file into jsonnet
    :param file_name: Jsonnet file
    :return:
    """
    json_parse = json.loads(_jsonnet.evaluate_file(file_name))

    return json_parse


def test_jsonnet_loads():
    """
    Testts loading jsonnet string. Examples from https://jsonnet.org/ref/bindings.html
    :return:
    """
    jsonnet_str = '''
    {
      person1: {
        name: "Alice",
        welcome: "Hello " + self.name + "!",
      },
      person2: self.person1 {
        name: std.extVar("OTHER_NAME"),
      },
    }
    '''

    ext_vars = {'OTHER_NAME': 'Bob'}

    json_item = jsonnet_loads(jsonnet_str, ext_vars=ext_vars)

    assert json_item["person1"]["welcome"] == "Hello Alice!"
    assert json_item["person2"]["name"] == "Bob"


if __name__ == "__main__":
    test_jsonnet_loads()