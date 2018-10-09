import json

from allennlp.models import load_archive
from allennlp.predictors import Predictor

from obqa.data.dataset_readers import ArcMultiChoiceWithFactsTextJsonReaderMultiSource
from obqa.models.qa.multi_choice.qa_multi_choice_know_reader_v1 import QAMultiChoiceKnowReader_v1
from obqa.predictors import Predictor_QA_MC_Knowledge_Visualize
from obqa.predictors.predictor_qa_mc_with_know_visualize import ID_TO_CHOICEKEY


def question_to_predictor_input(question):
    def get_text_from_tokens(tokens):
        clean_tokens = [x for x in tokens if not x.startswith("@") and not x.endswith("@")]

        return " ".join(clean_tokens)

    question_text = get_text_from_tokens(question["question_tokens"])
    choices = [get_text_from_tokens(x) for x in question["choice_tokens_list"]]
    facts_list = [get_text_from_tokens(x) for x in question["facts_tokens_list"]]
    input = {
        "question": question_text,
        "choice0": choices[0],
        "choice1": choices[1],
        "choice2": choices[2],
        "choice3": choices[3],
        "facts": "\n".join(facts_list),
    }

    return input

def predictor_input_to_pred_input_with_full_question_text(input):
    """
    Converts a standard parsed json to a text only format.
    :param input: Predictor json
    :return: Text-only question with choices and facts
    """
    new_input = {
        "question": input["question"] + " " + " ".join(["{0} {1}".format(ID_TO_CHOICEKEY[i], input["choice{0}".format(i)])
                                                  for i in range(10)
                                                  if "choice{0}".format(i) in input and i < len(ID_TO_CHOICEKEY)])
                    + "\nFacts:\n" + input["facts"]
    }

    return new_input


def test_predictor():
    question_json = {"id": "1700", "question_tokens": ["@start@", "For", "what", "does", "a", "stove", "generally", "generate", "heat", "?", "@end@"], "choice_tokens_list": [["@start@", "warming", "the", "air", "in", "the", "area", "@end@"], ["@start@", "heating", "nutrients", "to", "appropriate", "temperatures", "@end@"], ["@start@", "entertaining", "various", "visitors", "and", "guests", "@end@"], ["@start@", "to", "create", "electrical", "charges", "@end@"]], "facts_tokens_list": [["@start@", "UML", "can", "generate", "code", "@end@"], ["@start@", "generate", "is", "a", "synonym", "of", "beget", "@end@"], ["@start@", "Heat", "is", "generated", "by", "a", "stove", "@end@"], ["@start@", "A", "sonnet", "is", "generally", "very", "structured", "@end@"], ["@start@", "A", "fundamentalist", "is", "generally", "right", "-", "wing", "@end@"], ["@start@", "menstruation", "is", "generally", "crampy", "@end@"], ["@start@", "an", "erection", "is", "generally", "pleasurable", "@end@"], ["@start@", "gunfire", "is", "generally", "lethal", "@end@"], ["@start@", "ejaculating", "is", "generally", "pleasurable", "@end@"], ["@start@", "Huddersfield", "is", "generally", "urban", "@end@"], ["@start@", "warming", "is", "a", "synonym", "of", "calefacient", "@end@"], ["@start@", "heat", "is", "related", "to", "warming", "air", "@end@"], ["@start@", "a", "stove", "is", "for", "warming", "food", "@end@"], ["@start@", "an", "air", "conditioning", "is", "for", "warming", "@end@"], ["@start@", "The", "earth", "is", "warming", "@end@"], ["@start@", "a", "heat", "source", "is", "for", "warming", "up", "@end@"], ["@start@", "A", "foyer", "is", "an", "enterance", "area", "@end@"], ["@start@", "Being", "nosey", "is", "not", "appropriate", "@end@"], ["@start@", "seize", "is", "a", "synonym", "of", "appropriate", "@end@"], ["@start@", "a", "fitting", "room", "is", "used", "for", "something", "appropriate", "@end@"], ["@start@", "appropriate", "is", "a", "synonym", "of", "allow", "@end@"], ["@start@", "appropriate", "is", "similar", "to", "befitting", "@end@"], ["@start@", "appropriate", "is", "similar", "to", "grade", "-", "appropriate", "@end@"], ["@start@", "grade", "-", "appropriate", "is", "similar", "to", "appropriate", "@end@"], ["@start@", "A", "parlor", "is", "used", "for", "entertaining", "guests", "@end@"], ["@start@", "a", "back", "courtyard", "is", "for", "entertaining", "guests", "@end@"], ["@start@", "guest", "is", "a", "type", "of", "visitor", "@end@"], ["@start@", "a", "family", "room", "is", "for", "entertaining", "guests", "@end@"], ["@start@", "cooking", "a", "meal", "is", "for", "entertaining", "guests", "@end@"], ["@start@", "buying", "a", "house", "is", "for", "entertaining", "guests", "@end@"], ["@start@", "having", "a", "party", "is", "for", "entertaining", "guests", "@end@"], ["@start@", "a", "dining", "area", "is", "used", "for", "entertaining", "guests", "@end@"], ["@start@", "visitor", "is", "related", "to", "guest", "@end@"], ["@start@", "guest", "is", "related", "to", "visitor", "@end@"], ["@start@", "Electrical", "charges", "are", "additive", "@end@"], ["@start@", "Lightning", "is", "an", "electrical", "charge", "@end@"], ["@start@", "electrons", "have", "electrical", "charge", "@end@"], ["@start@", "A", "judge", "is", "in", "charge", "in", "a", "courtroom", "@end@"], ["@start@", "charge", "is", "a", "synonym", "of", "accusation", "@end@"], ["@start@", "A", "consultant", "can", "charge", "a", "fee", "to", "a", "client", "@end@"], ["@start@", "charge", "is", "a", "synonym", "of", "commission", "@end@"], ["@start@", "charge", "is", "a", "synonym", "of", "cathexis", "@end@"], ["@start@", "charge", "is", "not", "cash", "@end@"], ["@start@", "arraign", "entails", "charge", "@end@"], ["@start@", "a", "stove", "generates", "heat", "for", "cooking", "usually", "@end@"], ["@start@", "preferences", "are", "generally", "learned", "characteristics", "@end@"], ["@start@", "a", "windmill", "does", "not", "create", "pollution", "@end@"], ["@start@", "temperature", "is", "a", "measure", "of", "heat", "energy", "@end@"], ["@start@", "a", "hot", "something", "is", "a", "source", "of", "heat", "@end@"], ["@start@", "the", "moon", "does", "not", "contain", "water", "@end@"], ["@start@", "sunlight", "produces", "heat", "@end@"], ["@start@", "an", "oven", "is", "a", "source", "of", "heat", "@end@"], ["@start@", "a", "hot", "substance", "is", "a", "source", "of", "heat", "@end@"], ["@start@", "a", "car", "engine", "is", "a", "source", "of", "heat", "@end@"], ["@start@", "as", "the", "amount", "of", "rainfall", "increases", "in", "an", "area", ",", "the", "amount", "of", "available", "water", "in", "that", "area", "will", "increase", "@end@"], ["@start@", "sound", "can", "travel", "through", "air", "@end@"], ["@start@", "the", "greenhouse", "effect", "is", "when", "carbon", "in", "the", "air", "heats", "a", "planet", "'s", "atmosphere", "@end@"], ["@start@", "a", "community", "is", "made", "of", "many", "types", "of", "organisms", "in", "an", "area", "@end@"], ["@start@", "air", "is", "a", "vehicle", "for", "sound", "@end@"], ["@start@", "rainfall", "is", "the", "amount", "of", "rain", "an", "area", "receives", "@end@"], ["@start@", "an", "animal", "requires", "air", "for", "survival", "@end@"], ["@start@", "humidity", "is", "the", "amount", "of", "water", "vapor", "in", "the", "air", "@end@"], ["@start@", "if", "some", "nutrients", "are", "in", "the", "soil", "then", "those", "nutrients", "are", "in", "the", "food", "chain", "@end@"], ["@start@", "as", "heat", "is", "transferred", "from", "something", "to", "something", "else", ",", "the", "temperature", "of", "that", "something", "will", "decrease", "@end@"], ["@start@", "uneven", "heating", "causes", "convection", "@end@"], ["@start@", "as", "temperature", "during", "the", "day", "increases", ",", "the", "temperature", "in", "an", "environment", "will", "increase", "@end@"], ["@start@", "uneven", "heating", "of", "the", "Earth", "'s", "surface", "cause", "wind", "@end@"], ["@start@", "an", "animal", "needs", "to", "eat", "food", "for", "nutrients", "@end@"], ["@start@", "soil", "contains", "nutrients", "for", "plants", "@end@"], ["@start@", "if", "two", "objects", "have", "the", "same", "charge", "then", "those", "two", "materials", "will", "repel", "each", "other", "@end@"], ["@start@", "water", "is", "an", "electrical", "conductor", "@end@"], ["@start@", "a", "battery", "is", "a", "source", "of", "electrical", "energy", "@end@"], ["@start@", "metal", "is", "an", "electrical", "energy", "conductor", "@end@"], ["@start@", "when", "an", "electrical", "circuit", "is", "working", "properly", ",", "electrical", "current", "runs", "through", "the", "wires", "in", "that", "circuit", "@end@"], ["@start@", "brick", "is", "an", "electrical", "insulator", "@end@"], ["@start@", "wood", "is", "an", "electrical", "energy", "insulator", "@end@"], ["@start@", "a", "toaster", "converts", "electrical", "energy", "into", "heat", "energy", "for", "toasting", "@end@"]], "gold_label": 1, "gold_facts": {"fact1": "a stove generates heat for cooking usually", "fact2": "cooking involves heating nutrients to higher temperatures"}, "label_probs": [0.002615198493003845, 0.9686304330825806, 0.008927381597459316, 0.01982697658240795], "label_ranks": [3, 0, 2, 1], "predicted_label": 1, }

    inputs = question_to_predictor_input(question_json)
    inputs = predictor_input_to_pred_input_with_full_question_text(inputs)
    print(json.dumps(inputs, indent=4))

    archive = load_archive('_trained_models/model_CN5_1202.tar.gz')
    predictor = Predictor.from_archive(archive, 'predictor-qa-mc-with-know-visualize')

    result = predictor.predict_json(inputs)

    print(result)


if __name__ == "__main__":
    test_predictor()
