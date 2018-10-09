import json


class SimpleReaderKnowledgeFlexible:
    """
    Reads the fact depending on available fields. Currently switches conceptnet and DART.
    """

    def get_know_item_text(self, item):
        if "surfaceText" in item:
            return item["surfaceText"].replace("[[", "").replace("]]", "")
        elif "surfaceStart" in item:
            # conceptnet item
            return item["surfaceStart"] + " " + item["rel"] + " " + item["surfaceEnd"]
        elif "tkns" in item:
            return " ".join(item["tkns"])
        elif "SCIENCE-FACT" in item:
            return item["SCIENCE-FACT"]
        elif "fact_text" in item:
            return item["fact_text"]
        elif "Row Text" in item:
            return item["Row Text"]
        elif "Sentence" in item:  # Aristo Tuple KB v 5
            return item["Sentence"]
        else:
            print("Cannot handle item")
            return ""

    def get_reader_items_json_file(self, file_name):
        if file_name.endswith("csv"):
            for json_item in read_csv_file_to_json_flexible(file_name):
                yield json_item
        elif file_name.endswith("tsv"):
            for json_item in read_tsv_file_to_json_flexible(file_name):
                yield json_item
        else:
            for id, line in enumerate(open(file_name, mode="r")):
                json_item = json.loads(line.strip())

                yield json_item


def read_csv_file_to_json_flexible(file_name):
    """Reads a csv file to json
    See https://docs.python.org/3/library/csv.html for options and formats, etc.
    """
    import csv
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            yield row


def read_tsv_file_to_json_flexible(file_name):
    """Reads a tsv file to json
    See https://docs.python.org/3/library/csv.html for options and formats, etc.
    """
    import csv
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')

        for row in reader:
            yield row


if __name__ == "__main__":
    file_name = "tests/fixtures/data/knowledge/1202HITS.csv"

    read_csv_file_to_json_flexible(file_name)


