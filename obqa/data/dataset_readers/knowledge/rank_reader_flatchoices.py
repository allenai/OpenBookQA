import json
from obqa.utils.processing_utils import get_val_by_hier_key


class RankReader_FlatChoices_v1():
    """
    Reads knowledge facts from json and rankings for multi-choice (MC) QA
    """

    def read_facts(self, input_file):
        """
        Reads rankings from a json file.
        :param input_file: ranking json file
        :return: list of items with "text" key.
        """

        items = {}
        for l_id, line in enumerate(open(input_file, mode="r")):
            item = json.loads(line.strip())
            item_id = item["id"]
            item_id_hier = item_id.rsplit("__", 1)
            fact_weights = item["ext_fact_global_ids"]

            obj = items
            for ihid in range(len(item_id_hier)):
                hier_id = item_id_hier[ihid]
                if hier_id not in obj:
                    obj[hier_id] = {}
                obj = obj[hier_id]
            obj["weights"] = fact_weights

        return items

    def get_facts_text_with_weights_mask(self, loaded_ranks, all_know_facts, data_instance, max_facts):
        """
        Selects the knowledge text from a global list of facts, loadaded fact rankings and current data instance.
        :param loaded_ranks: A trie of "instance_id->ch_[choice_id]->weights"
        :param all_know_facts:  List of json facts that contain key "text"
        :param data_instance: Current question, choices instance
        :param max_facts: Max number of facts per field
        :return:
        """
        data_instance_id = data_instance["id"]
        choices_cnt = len(data_instance["question"]["choices"])

        all_facts_for_instance = {}

        fields_sparse_weights = []
        instance_trie = loaded_ranks[data_instance_id]

        fields_ids_to_check = ["q"] + ["ch_%s" % ch_id for ch_id in range(choices_cnt)]
        for fld_id in fields_ids_to_check:
            weights_for_ch = {x[0]: x[1] for x in
                              get_val_by_hier_key(instance_trie, [fld_id, "weights"], raise_key_error=False,
                                                  default=[])[:max_facts]}
            fields_sparse_weights.append(weights_for_ch)

            # update facts list
            for fact_id, weight in weights_for_ch.items():
                if fact_id in all_facts_for_instance:
                    all_facts_for_instance[fact_id] += 1
                else:
                    all_facts_for_instance[fact_id] = 1

        facts_by_freq = sorted([(k, freq) for k, freq in all_facts_for_instance.items()], key=lambda x: x[1],
                               reverse=True)
        fact_id_to_ordered_id_map = [kf[0] for oi, kf in enumerate(facts_by_freq)]

        instance_facts_list = [all_know_facts[fact_id]["text"] for fact_id, _ in facts_by_freq]

        # currently quesiton weights are ones!
        fields_weights_array = []
        for ch_sp_weight in fields_sparse_weights:
            non_sparse = [ch_sp_weight.get(fi, 0.0) for fi in fact_id_to_ordered_id_map]
            fields_weights_array.append(non_sparse)

        question_weights_array = fields_weights_array[0]
        choices_weights_array = fields_weights_array[1:] if len(fields_weights_array) > 1 else []

        return instance_facts_list, question_weights_array, choices_weights_array
