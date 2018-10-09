from overrides import overrides
from typing import Tuple

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

from obqa.models.qa.multi_choice import QAMultiChoiceKnowReader_v1


ID_TO_CHOICEKEY = ["(A)", "(B)", "(C)",
                   "(D)", "(E)", "(F)",
                   "(G)", "(H)", "(I)",
                   "(J)", "(K)", "(L)"]

def parse_question_text_with_chocies(question_full_str):
    """
    Concverts question with choices to a json used as input for the predictor
    :param question_full_str: Full question string ex. "What does conduct electriciy? (A) cotton shirt (B) suit of armor (C) wood toys (D) a glass"
    :return: A json with fields "question", "choice0", "choice1", ..
    """

    # split facts and question with choices
    facts_delim = "Facts:\n"
    question_with_facts = question_full_str.split(facts_delim)
    question_text = question_with_facts[0]
    facts_text = question_with_facts[1].strip() if len(question_with_facts) > 1 else ""

    # split choices
    fixed_delim = "@@choice_delim@@"
    for delim in ID_TO_CHOICEKEY:
        question_text = question_text.replace(delim, fixed_delim)

    question_split = question_text.split(fixed_delim)

    question_json = {}
    for i, txt in enumerate(question_split):
        if i == 0:
            question_json["question"] = txt.strip()
        else:
            question_json["choice{0}".format(i-1)] = txt.strip()

    question_json["facts"] = facts_text
    return question_json


@Predictor.register('predictor-qa-mc-with-know-visualize')
class Predictor_QA_MC_Knowledge_Visualize(Predictor):
    """Predictor wrapper for the obqa.models.QAMultiChoiceKnowReader_v1"""

    def __init__(self, model, dataset_reader):
        super().__init__(model, dataset_reader)

        model = self._model

        # model is expected to be QAMultiChoiceKnowReader_v1()
        if not isinstance(model, QAMultiChoiceKnowReader_v1):
            raise ValueError("model should be {0} but is {1}".format(" or ".join(["QAMultiChoiceKnowReader_v1",
                                                                                  "QAMultiChoiceKnowReader_v2",
                                                                                  "QAMultiChoiceKnowReader_v3"]),
                                                                     str(model)))

        model.return_question_to_choices_att = True
        model.return_question_to_facts_att = True
        model.return_choice_to_facts_att = True


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Converts the predictor input json to an Instance
        :param json_dict: Input json dict
        :return: Instance object with an additional json as meta data
        """
        question_text = json_dict['question']

        # parse the question text - in case that the question contains also choices and facts
        parsed_question = parse_question_text_with_chocies(question_text)

        # question text is always available in the parsed question
        question_text = parsed_question["question"]

        # get choices
        choice_text_list = []
        for choice_id, _ in enumerate(ID_TO_CHOICEKEY):
            choice_key = "choice{0}".format(choice_id)

            if choice_key in parsed_question:
                # check if the choices were contained already in the question text
                choice_text_list.append(parsed_question[choice_key])
            elif choice_key in json_dict:
                choice_text_list.append(json_dict[choice_key])
            else:
                # if the current choice ID is not found no more choices are expected
                break

        facts_text = ""
        if "facts" in parsed_question:
            facts_text = parsed_question["facts"]
        elif "facts" in json_dict:
            facts_text = json_dict["facts"]

        # get facts if any
        facts_text_list = [x.strip() for x in facts_text.split("\n")
                           if len(x.strip()) > 0 and not x.startswith("#")]

        # convert the input to Instance
        instance = self._dataset_reader.text_to_instance(item_id=0,
                         question_text=question_text,
                         choice_text_list=choice_text_list,
                         facts_text_list=facts_text_list,
                         question2facts_mapping=[0.0 for x in facts_text_list],
                         choice2facts_mapping=[[0.0 for x in facts_text_list] for xx in choice_text_list],
                         answer_id=0,
                         meta_fields=None
                        )

        # get labels to also pass to the predictor
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}