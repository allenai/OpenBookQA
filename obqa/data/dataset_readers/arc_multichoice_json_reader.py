from typing import Dict, List, Any
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from obqa.data.dataset_readers.common import token_indexer_dict_from_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("arc-multi-choice-json")
class ArcMultiChoiceJsonReader(DatasetReader):
    """
    Reading multi-choice QA instances in ARC format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:

        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 choice_value_type: str = None,
                 question_value_type: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._question_value_type = question_value_type

        self._choice_value_type = choice_value_type

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def get_question_text_from_item(self, item_json, question_value_type="question"):
        """
        Getting the ``question`` value depending on a configured question_value_type. Default is just question.
        The other settings are just hypothetical that can be used for experimenting with some oracle combinations.
        :param item_json: Item object
        :param question_value_type: The type of the question to form
        :return: String of the question
        """

        # this is the default setting
        question_text = item_json["question"]["stem"]

        # the options below can used for hypothetical oracle experiments
        if question_value_type == "fact1":
            question_text = item_json["fact1"]
        elif question_value_type == "fact2":
            question_text = item_json["fact2"]
        elif question_value_type == "fact1_fact2":
            question_text = item_json["fact1"] + " " + item_json["fact2"]
        elif question_value_type == "question_fact1":
            question_text = question_text + " " + item_json["fact1"]
        elif question_value_type == "question_fact2":
            question_text = question_text + " " + item_json["fact2"]
        elif question_value_type == "question_workerId":
            worker_id_token = "@%s@" % item_json["workerId"]
            question_text = worker_id_token + " " + question_text + " " + worker_id_token

        return question_text

    def get_choice_text_from_item(self, item_json, choice_id, choice_value_type="choice"):
        """
        Getting the ``choice`` value depending on a configured question_value_type. Default is just question.
        The other settings are just hypothetical that can be used for experimenting with some oracle combinations.
        :param item_json: Item object
        :param choice_id: The id of the choice to thet the choice text
        :param choice_value_type: Choice value type, Default is ``choice``
        :return: String of the choice
        """

        # this is the default setting
        choice_text = item_json["question"]["choices"][choice_id]["text"]

        # the options below can used for hypothetical oracle experiments
        if choice_value_type == "question_choice":
            question_text = item_json["question"]["stem"]
            choice_text = question_text + " " + choice_text
        elif choice_value_type == "choice_fact1":
            choice_text = choice_text + " " + item_json["fact1"]
        elif choice_value_type == "choice_fact2":
            choice_text = choice_text + " " + item_json["fact2"]
        elif choice_value_type == "choice_workerId":
            worker_id_token = "@%s@" % item_json["workerId"]
            choice_text = worker_id_token + " " + choice_text + " " + worker_id_token

        return choice_text

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as data_file:
            logger.info("Reading Multi-choice QA instances in ARC format from jsonl dataset at: %s", file_path)
            for curr_line_id, line in enumerate(data_file):
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question_text = self.get_question_text_from_item(item_json, self._question_value_type)

                choice_label_to_id = {}
                choice_text_list = []

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id

                    choice_text = self.get_choice_text_from_item(item_json, choice_id, self._choice_value_type)

                    choice_text_list.append(choice_text)

                answer_id = choice_label_to_id[item_json["answerKey"]]

                yield self.text_to_instance(item_id, question_text, choice_text_list, answer_id)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: Any,
                         question_text: str,
                         choice_text_list: List[str],
                         answer_id: int
                         ) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question_text)
        choices_tokens_list = [self._tokenizer.tokenize(x) for x in choice_text_list]
        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['choices_list'] = ListField([TextField(x, self._token_indexers) for x in choices_tokens_list])
        fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "choice_text_list": choice_text_list,
            "question_tokens": [x.text for x in question_tokens],
            "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ArcMultiChoiceJsonReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = token_indexer_dict_from_params(params.pop('token_indexers', {}))

        choice_value_type = params.get('choice_value_type', None)
        question_value_type = params.get('question_value_type', None)

        lazy = params.pop('lazy', False)

        return ArcMultiChoiceJsonReader(tokenizer=tokenizer,
                                        token_indexers=token_indexers,
                                        choice_value_type=choice_value_type,
                                        question_value_type=question_value_type,
                                        lazy=lazy)

    @classmethod
    def config_example(self):
        config_json = {
            "dataset_reader": {
                "type": "arc-multi-choice-json",
                "question_value_type": "question",
                "choice_value_type": "choice",
                "token_indexers": {
                    "tokens": {
                        "type": "single_id",
                        "lowercase_tokens": True
                    }
                },
                "tokenizers": {
                    "default": {
                        "start_tokens": ["@start@"],
                        "end_tokens": ["@end@"]
                    },
                },
            }
        }

        return json.dumps(config_json)
