import json

def safe_id(id):
    return id.replace("-", "_")

class ReaderARC_Question_Choice_Facts():
    """
    Reader for ARC QA dataset. It sets the question and choices to fact-ids to rank local facts.
    NOTE: To be used for ranking facts with know_reader_type "reader_gold_facts_arc_mc_qa"
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

    """
    def __init__(self, order="fact1_first"):
        self.order = order
        self.field_names = ["question", "choice"]
        self.id_field = "id"

    def get_reader_items_json_file(self, file_name):
        for id, line in enumerate(open(file_name, mode="r")):
            json_item = json.loads(line.strip())

            question_text = " ".join(["{0}_fact1".format(safe_id(json_item["id"])), 
                                      "{0}_fact1".format(safe_id(json_item["id"])),
                                      "{0}_fact2".format(safe_id(json_item["id"]))])

            curr_id = "{0}__q".format(json_item["id"])
            yield {"id": curr_id,
                   "question": question_text,
                   "choice": question_text,
                   "is_answer": False}

            for chi, choice in enumerate(json_item["question"]["choices"]):
                is_answer = choice["label"] == json_item["answerKey"]
                if is_answer:
                    choice_text = " ".join(["{0}_fact1".format(safe_id(json_item["id"])),
                                            "{0}_fact2".format(safe_id(json_item["id"])),
                                            "{0}_fact2".format(safe_id(json_item["id"]))])
                else:
                    choice_text = ""
                curr_id = "{0}__ch_{1}".format(json_item["id"], chi)
                yield {"id": curr_id,
                       "question": question_text,
                       "choice": choice_text,
                       "is_answer": is_answer}

    def get_reader_items_field_text_index(self, file_name):
        text_index = []
        items_list = []
        def update_index(field_value):
            text_index.append(field_value)
            return len(text_index) - 1

        for id, line in enumerate(open(file_name, mode="r")):
            json_item = json.loads(line.strip())

            if self.order == "fact1_first":
                question_text = " ".join(["{0}_fact1".format(safe_id(json_item["id"])),
                                          "{0}_fact1".format(safe_id(json_item["id"])),
                                          "{0}_fact2".format(safe_id(json_item["id"]))])
            else:
                question_text = " ".join(["{0}_fact1".format(safe_id(json_item["id"])),
                                          "{0}_fact2".format(safe_id(json_item["id"])),
                                          "{0}_fact2".format(safe_id(json_item["id"]))])

            question_text_id = update_index(question_text)

            curr_id = "{0}__q".format(json_item["id"])

            items_list.append({"id": curr_id,
                   "question": question_text_id,
                   "question_txt": question_text,
                   "choice": question_text_id,
                   "choice_text": "",
                   "is_answer": False})

            for chi, choice in enumerate(json_item["question"]["choices"]):
                is_answer = choice["label"] == json_item["answerKey"]
                if is_answer:
                    choice_text = " ".join(["{0}_fact1".format(safe_id(json_item["id"])),
                                            "{0}_fact2".format(safe_id(json_item["id"]))])
                else:
                    choice_text = ""
                curr_id = "{0}__ch_{1}".format(json_item["id"], chi)
                choice_text_id = update_index(choice_text)
                items_list.append( {"id": curr_id,
                       "question": question_text_id,
                       "question_text": question_text,
                       "choice": choice_text_id,
                       "choice_text": choice_text,
                       "is_answer": is_answer})

        return items_list, text_index