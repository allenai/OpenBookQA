import json


class ReaderFactsGoldFromArcMCQA:
    """
    Reads the fact depending on available fields. Currently switches conceptnet and DART.
    """

    def __init__(self, facts_limit=0):
        self._fact_limit = facts_limit

    def get_know_item_text(self, item):
        if "search_text" in item:
            return item["search_text"]
        else:
            print("Cannot handle item")
            return ""

    def get_reader_items_json_file(self, file_name):
        for id, line in enumerate(open(file_name, mode="r")):
            json_item = json.loads(line.strip())

            """
            This handles the facts from item:
            fact1, fact2 fields
            """
            def get_fact_from_qa_item(json_item, fact_name):
                fact_item = {}

                fact_item["id"] = "{0}_{1}".format(json_item["id"], fact_name)
                fact_item["fact_text"] = json_item[fact_name]
                fact_item["search_text"] = fact_item["id"].replace("-", "_")
                fact_item["meta"] = fact_name

                return fact_item

            has_more_facts = True
            curr_fact_id = 1
            while(has_more_facts and (self._fact_limit == 0 or curr_fact_id <= self._fact_limit)):
                fact_name = "fact{0}".format(curr_fact_id)
                if fact_name in json_item:
                    fact_item = get_fact_from_qa_item(json_item, fact_name)

                    yield fact_item
                else:
                    has_more_facts = False

                curr_fact_id += 1


if __name__ == "__main__":
    file_name = "/Users/todorm/research/data/allenai/2hopV3-rev1/full.jsonl"

    reader = ReaderFactsGoldFromArcMCQA(2)
    new_items = []
    for item in reader.get_reader_items_json_file(file_name):
        print(reader.get_know_item_text(item))


