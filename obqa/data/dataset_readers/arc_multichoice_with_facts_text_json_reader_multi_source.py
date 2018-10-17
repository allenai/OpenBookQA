from typing import Dict, List, Any
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
import numpy as np

from obqa.data.dataset_readers.common import read_cn5_surface_text_from_json, read_cn5_concat_subj_rel_obj_from_json, \
    read_json_flexible, KnowSourceManager, get_key_and_value_by_key_match, tokenizer_dict_from_params, \
    token_indexer_dict_from_params
from obqa.data.dataset_readers.knowledge.rank_reader_flatchoices import RankReader_FlatChoices_v1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

NO_RELEVANT_FACT_TEXT = "@norelevantfacts@"


@DatasetReader.register("arc-multi-choice-w-facts-txt-json-multi-source")
class ArcMultiChoiceWithFactsTextJsonReaderMultiSource(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
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
                 external_know_config: Params,
                 field_tokenizers: Dict[str, Tokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 choice_value_type: str = None,
                 question_value_type: str = None,
                 no_relevant_fact_add: bool = False,
                 no_relevant_fact_text: str = NO_RELEVANT_FACT_TEXT,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)

        self._field_tokenizers = field_tokenizers or {"default": WordTokenizer()}
        self._default_tokenizer = self._field_tokenizers.get("default")
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._external_know_config = external_know_config
        self._question_value_type = question_value_type

        self._choice_value_type = choice_value_type

        # add default fact that should be used as termination when no relevant facts are found.
        self._no_relevant_fact_add = no_relevant_fact_add
        self._no_relevant_fact_text = no_relevant_fact_text

        # knowledge sources
        key_to_reader = {
            "cn5-surface-text": read_cn5_surface_text_from_json,
            "cn5-concat-subj-rel-obj": read_cn5_concat_subj_rel_obj_from_json,
            "flexible-json": read_json_flexible
        }

        key_to_rank_reader = {
            "flat-q-ch-values-v1": RankReader_FlatChoices_v1(),
        }

        sources_usage = self._external_know_config.get("sources_use")
        self.knowledge_source_managers = []

        for ks_id, knowledge_source_config_single in enumerate(
                [x for i, x in enumerate(self._external_know_config.get("sources")) if sources_usage[i]]):
            know_source_manager = KnowSourceManager()
            know_source_manager._knowledge_source_config_single = knowledge_source_config_single  # currently only one source is supported
            know_source_manager._know_reader = key_to_reader[
                knowledge_source_config_single.get("type")]  # this one is a method used for loading knowledge file
            know_source_manager._know_rank_reader = key_to_rank_reader[knowledge_source_config_single.get(
                "rank_reader_type")]  # this one is class used for loading fact weights
            know_source_manager._use_know_cache = knowledge_source_config_single.get("max_facts_per_argument", False)
            know_source_manager._max_facts_per_argument = knowledge_source_config_single.get("max_facts_per_argument")

            know_source_manager._know_files_cache = {}
            know_source_manager._know_rank_files_cache = {}

            self.knowledge_source_managers.append(know_source_manager)

    def get_question_text_from_item(self, item_json, question_value_type):
        question_text = item_json["question"]["stem"]
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

    def get_choice_text_from_item(self, item_json, choice_id, choice_value_type):
        choice_text = item_json["question"]["choices"][choice_id]["text"]

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
        file_path_original = file_path
        file_path = cached_path(file_path)

        know_data_list = []
        know_rank_data_list = []
        # Prepare knowledge
        for ks_id, knowledge_source_manager in enumerate(self.knowledge_source_managers):
            # load external knowledge
            know_file_path = get_key_and_value_by_key_match(
                knowledge_source_manager._knowledge_source_config_single.get("dataset_to_know_json_file"),
                file_path_original, "any")
            know_rank_file_path = get_key_and_value_by_key_match(
                knowledge_source_manager._knowledge_source_config_single.get("dataset_to_know_rank_file"),
                file_path_original, "any")

            # facts
            know_data = []
            if know_file_path in knowledge_source_manager._know_files_cache:
                know_data = knowledge_source_manager._know_files_cache[know_file_path]
            else:
                know_data = knowledge_source_manager._know_reader(know_file_path)
                if knowledge_source_manager._use_know_cache:
                    knowledge_source_manager._know_files_cache[know_file_path] = know_data
            know_data_list.append(know_data)

            # ranking
            know_rank_data = []
            if know_rank_file_path in knowledge_source_manager._know_rank_files_cache:
                know_rank_data = knowledge_source_manager._know_files_cache[know_rank_file_path]
            else:
                know_rank_data = knowledge_source_manager._know_rank_reader.read_facts(know_rank_file_path)
                if knowledge_source_manager._use_know_cache:
                    knowledge_source_manager._know_files_cache[know_rank_file_path] = know_rank_data
            know_rank_data_list.append(know_rank_data)

        # Read knowledge facts to instances
        with open(file_path, 'r') as data_file:
            logger.info("Reading ARC instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question_text = self.get_question_text_from_item(item_json, self._question_value_type)

                gold_facts_text_meta = {"gold_facts":
                                            {"fact1": item_json.get("fact1", ""),
                                             "fact2": item_json.get("fact2", "")}
                                        }

                choice_label_to_id = {}
                choice_text_list = []

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id

                    choice_text = self.get_choice_text_from_item(item_json, choice_id, self._choice_value_type)
                    choice_text_list.append(choice_text)

                answer_id = choice_label_to_id[item_json["answerKey"]]

                # loading the facts from different sources
                facts_text_list = []
                question2facts_mapping = []
                choice2facts_mapping = None
                for ks_id, knowledge_source_manager in enumerate(self.knowledge_source_managers):
                    max_facts_per_field = knowledge_source_manager.get_max_facts_per_argument(file_path_original)
                    know_rank_data = know_rank_data_list[ks_id]
                    know_data = know_data_list[ks_id]
                    facts_text_list_curr, question2facts_mapping_curr, choice2facts_mapping_curr = \
                        knowledge_source_manager._know_rank_reader.get_facts_text_with_weights_mask(know_rank_data,
                                                                                                    know_data,
                                                                                                    item_json,
                                                                                                    max_facts_per_field)

                    facts_text_list.extend(facts_text_list_curr)
                    question2facts_mapping.extend(question2facts_mapping_curr)
                    if choice2facts_mapping is None:
                        choice2facts_mapping = choice2facts_mapping_curr
                    else:
                        for ch_id, ch_m in enumerate(choice2facts_mapping_curr):
                            choice2facts_mapping[ch_id].extend(ch_m)

                yield self.text_to_instance(item_id,
                                            question_text,
                                            choice_text_list,
                                            facts_text_list,
                                            question2facts_mapping,
                                            choice2facts_mapping,
                                            answer_id,
                                            gold_facts_text_meta)

    def tokenize(self, text, tokenizer_name="default"):
        tokenizer = self._field_tokenizers.get(tokenizer_name, self._default_tokenizer)
        return tokenizer.tokenize(text)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: Any,
                         question_text: str,
                         choice_text_list: List[str],
                         facts_text_list: List[str],
                         question2facts_mapping: List[float],
                         choice2facts_mapping: List[List[float]],
                         answer_id: int,
                         meta_fields: Dict = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        question_tokens = self.tokenize(question_text, "question")
        choices_tokens_list = [self.tokenize(x, "choice") for x in choice_text_list]
        facts_tokens_list = [self.tokenize(x, "fact") for x in facts_text_list]

        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['choices_list'] = ListField([TextField(x, self._token_indexers) for x in choices_tokens_list])
        fields['facts_list'] = ListField([TextField(x, self._token_indexers) for x in facts_tokens_list])
        fields['question2facts_map'] = ArrayField(np.asarray(question2facts_mapping))
        fields['choice2facts_map'] = ArrayField(np.asarray(choice2facts_mapping))

        fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "choice_text_list": choice_text_list,
            "facts_text_list": facts_text_list,
            "question_tokens": [x.text for x in question_tokens],
            "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
            "facts_tokens_list": [[x.text for x in ct] for ct in facts_tokens_list],
            "label_gold": answer_id,
        }

        if meta_fields is not None:
            for k, v in meta_fields.items():
                metadata[k] = v

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ArcMultiChoiceWithFactsTextJsonReaderMultiSource':
        # read tokenizers
        field_tokenizers = tokenizer_dict_from_params(params.get('tokenizers', {}))
        token_indexers = token_indexer_dict_from_params(params.get('token_indexers', {}))

        # external knowledge
        external_knowledge_params = params.pop('external_knowledge')

        choice_value_type = params.get('choice_value_type', None)
        question_value_type = params.get('question_value_type', None)

        no_relevant_fact_add = params.get('no_relevant_fact_add', False)
        no_relevant_fact_text = params.get('no_relevant_fact_text', NO_RELEVANT_FACT_TEXT)

        lazy = params.pop('lazy', False)
        # params.assert_empty(cls.__name__)

        return ArcMultiChoiceWithFactsTextJsonReaderMultiSource(field_tokenizers=field_tokenizers,
                                                                token_indexers=token_indexers,
                                                                external_know_config=external_knowledge_params,
                                                                choice_value_type=choice_value_type,
                                                                question_value_type=question_value_type,
                                                                no_relevant_fact_add=no_relevant_fact_add,
                                                                no_relevant_fact_text=no_relevant_fact_text,
                                                                lazy=lazy)

    @classmethod
    def config_example(self):
        config_json = {
            "dataset_reader": {
                "type": "arc-multi-choice-w-facts-txt-json",
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
                    }
                },
                "external_knowledge": {
                    "sources": [
                        {
                            "type": "flexible-json",
                            "name": "reads json file",
                            "use_cache": True,
                            "dataset_to_know_json_file": {"any": "/inputs_knowledge/knowledge.json"},
                            "dataset_to_know_rank_file": {"any": "/inputs_knowledge/full.jsonl.ranking.json"},
                            "rank_reader_type": "flat-q-ch-values-v1",
                            "max_facts_per_argument": 5
                        }
                    ],
                    "sources_use": [True]
                }
            },
        }

        return json.dumps(config_json, indent=4, sort_keys=True)
