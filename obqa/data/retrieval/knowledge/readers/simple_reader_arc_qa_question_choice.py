import json

import nltk


class SimpleReaderARC_Question_Choice():
    """
    Reader for ARC QA dataset:
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
    def __init__(self, raw_field_for_question="question", raw_field_for_choice_supporting_fact="question"):
        self.raw_data_field_for_question = raw_field_for_question
        self.raw_field_for_choice_supporting_fact = raw_field_for_choice_supporting_fact
        self.field_names = ["question", "choice"]
        self.id_field = "id"

    def get_reader_items_json_file(self, file_name):
        for id, line in enumerate(open(file_name, mode="r")):
            json_item = json.loads(line.strip())

            # question text
            question_text_raw = json_item["question"]["stem"]
            if self.raw_data_field_for_question != "question":
                question_text_raw = json_item[self.raw_data_field_for_question]

            question_text = question_text_raw.lower()

            # choice supporting fact
            choice_supporting_fact_text_raw = json_item["question"]["stem"]
            if self.raw_field_for_choice_supporting_fact == "fact1_plus_question":
                choice_supporting_fact_text_raw = json_item["question"]["stem"] + " " + json_item["fact1"]
            elif self.raw_field_for_choice_supporting_fact == "fact1_diff_question":
                question_tokens = set(nltk.word_tokenize(json_item["question"]["stem"].lower()))
                fact1_tokens = nltk.word_tokenize(json_item["fact1"].lower())

                diff = [tkn for tkn in fact1_tokens if tkn not in question_tokens]
                choice_supporting_fact_text_raw = " ".join(diff)
            elif self.raw_field_for_choice_supporting_fact != "question":
                choice_supporting_fact_text_raw = json_item[self.raw_field_for_choice_supporting_fact]

            choice_supporting_fact_text = choice_supporting_fact_text_raw.lower()

            curr_id = "{0}__q".format(json_item["id"])
            yield {"id": curr_id,
                   "question": question_text,
                   "choice": question_text,
                   "is_answer": False}

            for chi, choice in enumerate(json_item["question"]["choices"]):
                choice_text = choice["text"].lower()
                curr_id = "{0}__ch_{1}".format(json_item["id"], chi)
                yield {"id": curr_id,
                       "question": choice_supporting_fact_text,
                       "choice": choice_text,
                       "is_answer": choice["label"] == json_item["answerKey"]}

    def get_reader_items_field_text_index(self, file_name):
        text_index = []
        items_list = []
        def update_index(field_value):
            text_index.append(field_value)
            return len(text_index) - 1

        for id, line in enumerate(open(file_name, mode="r")):
            try:
                json_item = json.loads(line.strip())

                # question text
                question_text_raw = json_item["question"]["stem"]
                if self.raw_data_field_for_question != "question":
                    question_text_raw = json_item[self.raw_data_field_for_question]

                question_text = question_text_raw.lower()
                question_text_id = update_index(question_text)

                # supporting fact
                # choice supporting fact
                choice_supporting_fact_text_raw = json_item["question"]["stem"]
                if self.raw_field_for_choice_supporting_fact == "fact1_plus_question":
                    choice_supporting_fact_text_raw = json_item["question"]["stem"] + " " + json_item["fact1"]
                elif self.raw_field_for_choice_supporting_fact == "fact1_diff_question":
                    question_tokens = set(nltk.word_tokenize(json_item["question"]["stem"].lower()))
                    fact1_tokens = nltk.word_tokenize(json_item["fact1"].lower())

                    diff = [tkn for tkn in fact1_tokens if tkn not in question_tokens]
                    choice_supporting_fact_text_raw = " ".join(diff)
                elif self.raw_field_for_choice_supporting_fact != "question":
                    choice_supporting_fact_text_raw = json_item[self.raw_field_for_choice_supporting_fact]

                choice_supporting_fact_text = choice_supporting_fact_text_raw.lower()
                choice_supporting_fact_text_id = update_index(choice_supporting_fact_text)

                curr_id = "{0}__q".format(json_item["id"])

                items_list.append({"id": curr_id,
                       "question": question_text_id,
                       "question_txt": question_text,
                       "choice": question_text_id,
                       "choice_text": "",
                       "is_answer": False})

                for chi, choice in enumerate(json_item["question"]["choices"]):
                    choice_text = choice["text"].lower()
                    curr_id = "{0}__ch_{1}".format(json_item["id"], chi)
                    choice_text_id = update_index(choice_text)
                    items_list.append( {"id": curr_id,
                           "question": choice_supporting_fact_text_id,
                           "question_text": choice_supporting_fact_text,
                           "choice": choice_text_id,
                           "choice_text": choice_text,
                           "is_answer": choice["label"] == json_item["answerKey"]})
            except Exception as e:
                print("Error on file line %s" % id)
                raise e

        return items_list, text_index