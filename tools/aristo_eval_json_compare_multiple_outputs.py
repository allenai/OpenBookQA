"""
This scripts compares the output of multiple models by exporting the predictions to csv
"""

# QA MC evaluator output:
# {"id": "7-623", "question_tokens": ["@start@", "If", "someone", "is", "dying", "of", "thirst", ",", "they", "can", "hydrate", "by", "@end@"], "choice_tokens_list": [["@start@", "drink", "acid", "@end@"], ["@start@", "visiting", "a", "valley", "@end@"], ["@start@", "find", "snow", "@end@"], ["@start@", "catching", "rain", "@end@"]], "facts_tokens_list": [["@start@", "thirstiness", "is", "a", "synonym", "of", "thirst", "@end@"], ["@start@", "My", "thirst", "is", "unquenched", "@end@"], ["@start@", "moisturizers", "are", "used", "to", "hydrate", "@end@"], ["@start@", "dying", "is", "for", "recoloring", "@end@"], ["@start@", "dying", "is", "a", "synonym", "of", "anxious", "@end@"], ["@start@", "acid", "is", "a", "synonym", "of", "sulfurous", "@end@"], ["@start@", "Cimetidine", "is", "an", "acid", "reducer", "@end@"], ["@start@", "orthophosphorous", "acid", "is", "a", "synonym", "of", "hypophosphorous", "acid", "@end@"], ["@start@", "orthophosphoric", "acid", "is", "a", "synonym", "of", "phosphoric", "acid", "@end@"], ["@start@", "orthoboric", "acid", "is", "a", "synonym", "of", "boric", "acid", "@end@"], ["@start@", "a", "parlor", "is", "used", "for", "visiting", "@end@"], ["@start@", "A", "dale", "is", "a", "broad", "valley", "@end@"], ["@start@", "a", "valley", "is", "for", "separating", "moutains", "@end@"], ["@start@", "We", "can", "see", "the", "whole", "valley", "from", "here", "@end@"], ["@start@", "Ruhr", "Valley", "is", "a", "synonym", "of", "Ruhr", "@end@"], ["@start@", "*", "Something", "you", "find", "in", "Antrarctica", "is", "snow", "@end@"], ["@start@", "*", "Something", "you", "find", "on", "the", "ground", "is", "snow", "@end@"], ["@start@", "You", "are", "likely", "to", "find", "water", "in", "snow", "@end@"], ["@start@", "You", "are", "likely", "to", "find", "snow", "in", "the", "mountains", "@end@"], ["@start@", "Snow", "is", "a", "synonym", "of", "Baron", "Snow", "of", "Leicester", "@end@"], ["@start@", "catching", "is", "a", "synonym", "of", "transmittable", "@end@"], ["@start@", "catching", "mumps", "is", "for", "being", "infected", "@end@"], ["@start@", "a", "projectile", "is", "for", "catching", "@end@"], ["@start@", "a", "mitt", "is", "used", "for", "catching", ".", "@end@"], ["@start@", "Sometimes", "catching", "mumps", "causes", "being", "discomfortable", "@end@"], ["@start@", "rain", "is", "a", "source", "of", "water", "@end@"], ["@start@", "rain", "is", "something", "you", "can", "drink", "@end@"]], "gold_label": 3, "label_probs": [0.0004977978533133864, 9.097841591199085e-09, 8.220109157264233e-05, 0.9994199872016907], "label_ranks": [1, 3, 2, 0], "predicted_label": 3, "attentions": {"att_q_to_ch": {"ctx__ctx": [0.22003990411758423, 0.054642874747514725, 0.18165147304534912, 0.5436657667160034], "ctx+kn__ctx": [0.2538149654865265, 0.07607488334178925, 0.16426706314086914, 0.5058431029319763], "ctx__ctx+kn": [0.21901708841323853, 0.10258235037326813, 0.2789818048477173, 0.39941874146461487], "ctx+kn__ctx+kn": [0.2368510216474533, 0.1347828358411789, 0.22711627185344696, 0.40124988555908203], "kn__kn": [0.1934242844581604, 0.3064635694026947, 0.241935595870018, 0.2581765055656433], "kn__ctx": [0.3425443172454834, 0.07934822887182236, 0.12832899391651154, 0.4497784674167633], "ctx__kn": [0.23279854655265808, 0.17660053074359894, 0.28760382533073425, 0.3029971122741699], "final": [0.0004977978533133864, 9.097841591199085e-09, 8.220109157264233e-05, 0.9994199872016907]}, "att_q_to_f": {"src1": [0.0025196450296789408, 0.000950635236222297, 0.03018159419298172, 0.002228266093879938, 0.015374170616269112, 0.019589293748140335, 0.005597833078354597, 0.1457756757736206, 0.037033822387456894, 0.24058791995048523, 0.0022169980220496655, 0.00043176396866329014, 0.0004479088820517063, 0.0029872425366193056, 0.0003698126820381731, 0.06536025553941727, 0.1305832862854004, 0.08518018573522568, 0.024547619745135307, 0.009095474146306515, 0.0012500130105763674, 0.034235142171382904, 0.005543697625398636, 0.06025818735361099, 0.006290045101195574, 0.018302779644727707, 0.05306074768304825]}, "att_ch_to_f": {"src1": [[0.00039694661973044276, 0.00011020875535905361, 0.019029241055250168, 0.0002876016078516841, 0.0010408380767330527, 0.03954869508743286, 0.030369630083441734, 0.11328822374343872, 0.10934063792228699, 0.6425564289093018, 0.0001902178191812709, 0.00014034024206921458, 8.423704275628552e-05, 0.00015886686742305756, 7.473044388461858e-05, 0.0014520646072924137, 0.001293452805839479, 0.0033453444484621286, 0.00021372165065258741, 0.0005627079517580569, 0.0001798375160433352, 0.0016555507900193334, 0.000356712203938514, 0.002601429121568799, 0.00031341463909484446, 0.026545409113168716, 0.004863531328737736], [0.0044834245927631855, 0.0017755868611857295, 0.0030517259147018194, 0.0006117151933722198, 0.011938047595322132, 0.017409492284059525, 0.002923798281699419, 0.03983985632658005, 0.010166068561375141, 0.09742254763841629, 0.009745056740939617, 0.3004915118217468, 0.03921127691864967, 0.023953694850206375, 0.07409989833831787, 0.037365809082984924, 0.0789058655500412, 0.027727318927645683, 0.04528271034359932, 0.05158158391714096, 0.004708438646048307, 0.012876606546342373, 0.0039002476260066032, 0.07115103304386139, 0.0012306967983022332, 0.023315098136663437, 0.004830938298255205], [0.000735534995328635, 0.0008231538231484592, 0.005404084920883179, 0.0013377502327784896, 0.005691652186214924, 0.006880020257085562, 0.0012061892775818706, 0.006810392253100872, 0.0015618063043802977, 0.018648652359843254, 0.00493050878867507, 0.004048923961818218, 0.002081406768411398, 0.00410253182053566, 0.00046783461584709585, 0.2711031436920166, 0.24387383460998535, 0.18112796545028687, 0.04848222807049751, 0.048853158950805664, 0.002262946916744113, 0.01579313352704048, 0.007902363315224648, 0.0445905476808548, 0.0037637108471244574, 0.05588972568511963, 0.011626837775111198], [0.0007800919702276587, 0.0016053472645580769, 0.00722577515989542, 0.0019369872752577066, 0.007749153301119804, 0.018388807773590088, 0.002986151957884431, 0.056029800325632095, 0.0044541084207594395, 0.04652535170316696, 0.003279020544141531, 0.0038416299503296614, 0.0027349628508090973, 0.0016861994517967105, 0.0004051118448842317, 0.024296386167407036, 0.03763830289244652, 0.029681244865059853, 0.005087527446448803, 0.01747681386768818, 0.019957931712269783, 0.1894281953573227, 0.056192584335803986, 0.20039379596710205, 0.026488522067666054, 0.19715982675552368, 0.03657037392258644]]}}, "know_inter_weights": {"ctx__ctx": 2.075143814086914, "ctx+kn__ctx": 2.077263116836548, "ctx__ctx+kn": 2.3152270317077637, "ctx+kn__ctx+kn": 2.391340494155884, "kn__kn": 2.0042362213134766, "kn__ctx": 1.8845239877700806, "ctx__kn": 2.092815399169922}}
# aristo evaluator input:
#{"id": "Mercury_SC_407695", "prediction": {"stem": "Juan and LaKeisha roll a few objects down a ramp. They want to see which object rolls the farthest. What should they do so they can repeat their investigation?", "choices": [{"text": "Put the objects in groups.", "label": "A", "score": 0, "support": ""}, {"text": "Change the height of the ramp.", "label": "B", "score": 0.0266, "support": {"text": "Examples of Inclined Planes Ramp a ramp is used to push up an object or roll it down.", "type": "sentence", "ir_pos": 1, "ir_score": 55.234158}}, {"text": "Choose different objects to roll.", "label": "C", "score": 0.4293, "support": {"text": "An object is rolling down an incline.", "type": "sentence", "ir_pos": 1, "ir_score": 57.172306}}, {"text": "Record the details of the investigation.", "label": "D", "score": 0, "support": ""}]}, "answerKey": "D", "selected_answers": "C", "question_score": 0}
import json
import numpy as np
import os
import sys
import argparse
import csv

id2key = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def read_jsonl(file_name):
    json_items = []
    for line in open(file_name, mode="r"):
        json_item = json.loads(line.strip())
        json_items.append(json_item)

    return json_items

def jsonlist_to_dict(json_list, key_field):
    dict = {}
    for item in json_list:
        dict[item[key_field]] = item

    return dict

# gold
# {
#   "annotation": "Y",
#   "answerKey": "A",
#   "clarity": "1.80",
#   "fact1": "deep sea animals live deep in the ocean",
#   "fact2": "Examples of deep sea animals are angler fish and frilled sharks",
#   "humanScore": "0.80",
#   "id": "8-376",
#   "question": {
#     "choices": [
#       {
#         "label": "A",
#         "text": "Deep sea animals"
#       },
#       {
#         "label": "B",
#         "text": "fish"
#       },
#       {
#         "label": "C",
#         "text": "Long Sea Fish"
#       },
#       {
#         "label": "D",
#         "text": "Far Sea Animals"
#       }
#     ],
#     "stem": "Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as"
#   },
#   "workerId": "A2R0YYUAWNT7UD"
# }
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, help='<Required> Gold file in ARC format', required=True)
    parser.add_argument('--files', type=str, nargs='+')
    parser.add_argument('--friendly_names', nargs='+', help='List of run names. Fields are in Aristo Evaluator format', required=False, default=None)
    parser.add_argument('--out', type=str, help='<Required> Output file', required=True)

    args = parser.parse_args()

    gold_arc_json_list = read_jsonl(args.gold)

    predictions_dict_json_list = []

    friendly_names = []
    if not args.friendly_names or args.friendly_names is None:
        friendly_names = [os.path.basename(f_name) for f_name in args.file]
    else:
        friendly_names = args.friendly_names

    assert len(friendly_names) == len(args.files), "Num of friendly_names and files does not match"
    for fid, f_name in enumerate(args.files):
        curr_json_list = jsonlist_to_dict(read_jsonl(f_name), "id")
        predictions_dict_json_list.append(curr_json_list)


    max_num_choices = max([len(x["question"]["choices"]) for x in gold_arc_json_list])

    assert max_num_choices < len(id2key), "max_num_choices should be <= %d" % len(id2key)
    out_fields_names = ["id", "question"] \
                       + [id2key[chid] for chid in range(max_num_choices)]

    if "fact1" in gold_arc_json_list[0]:
        out_fields_names = out_fields_names + ["fact1"]

    if "fact2" in gold_arc_json_list[0]:
        out_fields_names = out_fields_names + ["fact2"]

    out_fields_names = out_fields_names \
                       + ["answer"] \
                       + ["%s_gold" % pred_name for pred_name in friendly_names]\
                       + ["%s_gold_score" % pred_name for pred_name in friendly_names]\
                       + ["%s_pred" % pred_name for pred_name in friendly_names]\
                       + ["%s_pred_score" % pred_name for pred_name in friendly_names]\
                       + ["%s_all_scores" % pred_name for pred_name in friendly_names]


    with open(args.out, mode="w") as output:
        output_csv = csv.DictWriter(output, fieldnames=out_fields_names, quoting=csv.QUOTE_NONNUMERIC)
        output_csv.writeheader()
        for item in gold_arc_json_list:
            # metadata
            out_item_json = {"id": item["id"],
                             "question": item["question"]["stem"],
                             "answer": item["answerKey"]
                             }

            gold_answer = out_item_json["answer"]

            # OpenBookQA fields
            if "fact1" in out_fields_names:
                out_item_json["fact1"] = item.get("fact1", "")

            if "fact2" in out_fields_names:
                out_item_json["fact2"] = item.get("fact2", "")

            # default choices
            for chid in range(max_num_choices):
                out_item_json[id2key[chid]] = ""

            # set actual chocies text
            for choice in item["question"]["choices"]:
                out_item_json[choice["label"]] = choice["text"]

            # predictions
            for pred_id, pred_name in enumerate(friendly_names):
                predictions_dict_curr = predictions_dict_json_list[pred_id]
                prediction_curr_quesiton = predictions_dict_curr[out_item_json["id"]]

                best_score = - float("inf")
                best_choice = None

                gold_score = 0.0
                all_scores = ""
                for choice in prediction_curr_quesiton["prediction"]["choices"]:
                    all_scores += "{0}) {1:0.4f} |".format(choice["label"], choice["score"])
                    if choice["score"] > best_score:
                        best_score = choice["score"]
                        best_choice = choice

                    if choice["label"] == gold_answer:
                        gold_score = choice["score"]

                best_choice_label = best_choice["label"]

                out_item_json["%s_gold" % pred_name] = 1 if best_choice_label == gold_answer else 0
                out_item_json["%s_gold_score" % pred_name] = gold_score
                out_item_json["%s_pred" % pred_name] = best_choice_label
                out_item_json["%s_pred_score" % pred_name] = best_score
                out_item_json["%s_all_scores" % pred_name] = all_scores

            output_csv.writerow(out_item_json)

    # new_json_item = {
    #     "id": orig_json_item["id"],
    #     "prediction": {"choices": [{"label": id2key[id], "score": score if not reverse_probs else 1 - score} for id, score in enumerate(orig_json_item["label_probs"])]}
    # }

