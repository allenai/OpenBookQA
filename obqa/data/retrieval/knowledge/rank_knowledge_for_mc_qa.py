import codecs
import json
import logging

import os

import argparse
import time

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

from obqa.data.retrieval.knowledge.readers.reader_arc_qa_question_choice_facts import ReaderARC_Question_Choice_Facts
from obqa.data.retrieval.knowledge.readers.reader_gold_facts_arc_mc_qa import ReaderFactsGoldFromArcMCQA
from obqa.data.retrieval.knowledge.readers.simple_reader_arc_qa_question_choice import SimpleReaderARC_Question_Choice
from obqa.data.retrieval.knowledge.readers.simple_reader_knowledge_flexible import SimpleReaderKnowledgeFlexible

try:
    from spacy.lang.en.stop_words import STOP_WORDS
except:
    from spacy.en import STOP_WORDS

comb_funcs = {
    "mul": np.multiply,
    "add": np.add,
    "2x+y": lambda x, y: 2 * x + y
}


def get_similarities(query_feats, para_features, top=10, combine_feat_scores="mul"):
    """
    Get similarities based on multiple independent queries that are then combined using combine_feat_scores
    :param query_feats: Multiple vectorized text queries
    :param para_features: Multiple vectorized text paragraphs that will be scored against the queries
    :param top: Top N facts to keep
    :param combine_feat_scores: The way for combining the multiple scores
    :return: Ranked fact ids with scores List[tuple(id, weight)]
    """
    scores_per_feat = [pairwise_distances(q_feat, para_features, "cosine").ravel() for q_feat in query_feats]  # this is distance - low is better!!!
    comb_func = comb_funcs[combine_feat_scores]

    smoothing_val = 0.000001
    max_val = pow((1 + smoothing_val), 2)
    dists = scores_per_feat[0] + smoothing_val
    if len(scores_per_feat) > 1:
        for i in range(1, len(scores_per_feat)):
            dists = comb_func(scores_per_feat[i] + smoothing_val, dists)
    sorted_ix = np.argsort(dists).tolist()  # this is asc (lowers first), in case of ties, uses the earlier paragraph

    return [[i, (max_val - dists[i]) / max_val] for i in sorted_ix][:top]


def combine_similarities(scores_per_feat, top=10, combine_feat_scores="mul"):
    """
    Get similarities based on multiple independent queries that are then combined using combine_feat_scores
    :param query_feats: Multiple vectorized text queries
    :param para_features: Multiple vectorized text paragraphs that will be scored against the queries
    :param top: Top N facts to keep
    :param combine_feat_scores: The way for combining the multiple scores
    :return: Ranked fact ids with scores List[tuple(id, weight)]
    """
    # scores_per_feat = [pairwise_distances(q_feat, para_features, "cosine").ravel() for q_feat in query_feats]  # this is distance - low is better!!!
    comb_func = comb_funcs[combine_feat_scores]

    smoothing_val = 0.000001
    max_val = pow((1 + smoothing_val), 2)
    dists = scores_per_feat[0] + smoothing_val
    if len(scores_per_feat) > 1:
        for i in range(1, len(scores_per_feat)):
            dists = comb_func(scores_per_feat[i] + smoothing_val, dists)
    sorted_ix = np.argsort(dists).tolist()  # this is asc (lowers first) ,in case of ties, uses the earlier paragraph

    max_val = max(np.max(dists), 1)
    return [[i, (max_val - dists[i]) / max_val] for i in sorted_ix][:top]


def get_similarities_ext(query_feats, para_features, top=0, combine_feat_scores="mul", default_weights=None):
    """
    Get similarities based on multiple independent queries that are then combined using combine_feat_scores
    :param query_feats: Multiple vectorized text queries
    :param para_features: Multiple vectorized text paragraphs that will be scored against the queries
    :param top: Top N facts to keep
    :param combine_feat_scores: The way for combining the multiple scores. mul, avg, sort supported.
    :param default_weights: Default weights for each paragraph in para_features. This will be used as first value to sort by when "sort" is used for combine_feat_scores.
    :return: Ranked fact ids with scores List[tuple(id, weight)]
    """
    scores_per_feat = [pairwise_distances(q_feat, para_features, "cosine").ravel() for q_feat in query_feats]

    if combine_feat_scores == "sort":
        if default_weights is not None:
            scores_per_feat.append(default_weights)
        sorted_ix = np.lexsort(tuple(scores_per_feat))
    else:
        dists = scores_per_feat[0]
        if len(scores_per_feat) > 1:
            comb_func = np.multiply if combine_feat_scores == "mul" else np.add
            for i in range(1, len(scores_per_feat)):
                dists = comb_func(scores_per_feat[i], dists)

        sorted_ix = np.argsort(dists)  # in case of ties, use the earlier paragraph

    res = [[i, 1.0 - scores_per_feat[0][i]] for i in sorted_ix]
    if top > 0:
        return res[:top]
    else:
        return res


def print_json_formatted(json_item):
    print(json.dumps(json_item, indent=4, sort_keys=True))


def process_item_dart(text, add_entity=False):
    """Processes dart dataset"""
    tokens = text.split()
    tokens_filtered = [t for t in tokens if not (len(t) >= 2 and t[0].isupper() and not t[1].isupper())]
    if add_entity and len(tokens_filtered) < len(tokens):
        tokens_filtered.extend(["person", "organization"])
    return " ".join(tokens_filtered)


knowledge_reader_types = {
    "simple_reader_knowledge_flexible": SimpleReaderKnowledgeFlexible(),
    "reader_gold_facts_arc_mc_qa_1": ReaderFactsGoldFromArcMCQA(1),
    "reader_gold_facts_arc_mc_qa_2": ReaderFactsGoldFromArcMCQA(2)
}

data_reader_types = {
    "simple_reader_arc_qa_question_choice": SimpleReaderARC_Question_Choice("question"),
    "simple_reader_arc_qa_fact1_choice": SimpleReaderARC_Question_Choice("fact1"),
    "simple_reader_arc_qa_fact2_choice": SimpleReaderARC_Question_Choice("fact2"),
    "simple_reader_arc_qa_fact1_choice_and_q": SimpleReaderARC_Question_Choice("question", "fact1"),
    # fact1 with choices, question
    "simple_reader_arc_qa_fact1_plus_question_choice_and_q": SimpleReaderARC_Question_Choice("question",
                                                                                             "fact1_plus_question"),
    # fact1+question with choices, question
    "simple_reader_arc_qa_fact1_diff_question_choice_and_q": SimpleReaderARC_Question_Choice("question",
                                                                                             "fact1_diff_question"),
    # fact1-question with choices, question
    "reader_arc_qa_question_choice_facts": ReaderARC_Question_Choice_Facts("fact1_first"),
    "reader_arc_qa_question_choice_facts_fact2_first": ReaderARC_Question_Choice_Facts("fact2_first")
}

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]


if __name__ == "__main__":
    # Sample run
    # Set logging info
    logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s :  %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Enable file logging
    # logFileName = '%s/%s-%s.log' % ('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
    # fileHandler = logging.FileHandler(logFileName, 'wb')
    # fileHandler.setFormatter(logFormatter)
    # logger.addHandler(fileHandler)

    # Enable console logging
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(consoleHandler)

    parser = argparse.ArgumentParser(
        description='Given dataset and knowledge file, ranks knowledge for the data triples')
    parser.add_argument('-o', '--out_dir', dest="out_dir", metavar='your/out/dir', type=str,
                        help='The name of the run. Very short name to be refered to for this configuration..')

    parser.add_argument('-i', '--input_files', dest="input_files", metavar='file1;file2;file3;', type=str,
                        help='Paths to files to extract knowledge for.')

    parser.add_argument('--dataset_reader', dest="dataset_reader", choices=data_reader_types.keys(), type=str,
                        help='Reader type to load the dataset to retrieve knowledge for',
                        default="simple_reader_arc_qa_question_choice")

    parser.add_argument('--knowledge_reader', dest="knowledge_reader", choices=knowledge_reader_types.keys(), type=str,
                        help='Reader type to load the dataset to retrieve knowledge for',
                        default="simple_reader_knowledge_flexible")

    parser.add_argument('-k', '--knowledge_file', dest="knowledge_file", metavar='path/to/knowledge_file.json',
                        type=str,
                        help='Path to the knowledge file. This is a json file. Currently Concepnet5 Json is supported.')

    parser.add_argument('-n', '--run_name', dest="run_name", metavar='cn5_tfidf_quest_plus_choice', type=str,
                        help='The name of the run. Very short name to be refered to for this configuration..')

    parser.add_argument('--max_facts_per_choice', dest="max_facts_per_choice", metavar='N', type=int,
                        help='The number of facts per choice item.')

    parser.add_argument('--limit_items', dest="limit_items", metavar='N', type=int, default=0,
                        help='This reads only some number of N items. To be used for debug purposes')

    args = parser.parse_args()
    for k, v in args.__dict__.items():
        logging.info("{0}:{1}".format(k, v))

    out_directory = args.out_dir
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
        logging.info("Created directory %s" % out_directory)

    input_files = args.input_files
    knowledge_file_json = args.knowledge_file  # "/Users/todorm/research/data/knowledge/conceptnet5_surfacetext_limit_0.json"
    know_run_name = args.run_name  # "cn5_tfidf_quest_plus_choice"

    max_facts_per_choice = args.max_facts_per_choice
    limit_for_debug = args.limit_items

    combine_feat_scores = "mul"  # sum, mul, 2x+y
    data_utils = data_reader_types[args.dataset_reader]
    know_utils = knowledge_reader_types[args.knowledge_reader]
    logging.info("out_directory:%s" % out_directory)
    logging.info("input data files:%s" % input_files)
    logging.info("dataset_reader:%s" % args.dataset_reader)
    logging.info("knowledge_file_json:%s" % knowledge_file_json)
    logging.info("knowledge_reader:%s" % args.knowledge_reader)
    logging.info("know_run_name:%s" % know_run_name)
    logging.info("max_facts_per_choice:%s" % max_facts_per_choice)

    know_fact_ranking_out_file = "{0}/knowledge.json".format(out_directory).replace("//", "/")

    # LOAD DATA
    start = time.time()
    logging.info("Loading data...")
    paragraphs_items = []

    # read knowledge and writes a separate file for the current setup
    know_file_out = codecs.open(know_fact_ranking_out_file, mode="w", encoding="utf-8")

    for lid, item in enumerate(know_utils.get_reader_items_json_file(knowledge_file_json)):
        know_file_out.write(json.dumps(item))
        know_file_out.write("\n")
        curr_par = know_utils.get_know_item_text(item)
        paragraphs_items.append((lid, curr_par))

    know_file_out.close()

    logging.info("Readable knowledge saved to %s" % know_fact_ranking_out_file)
    par_ids, paragraphs = zip(*paragraphs_items)
    logging.info("%s paragraphs pre-processed in %s" % (len(paragraphs), time.time() - start))

    # process
    file_names_list = input_files.split(";")
    for file_name in file_names_list:
        logging.info("Ranking data for %s" % file_name)

        file_name_base = os.path.basename(file_name)
        # data_knowledge_ranking_file_name = "{0}/{1}.{2}.ranking.json".format(out_directory, file_name_base, know_run_name)
        data_knowledge_ranking_file_name = "{0}/{1}.ranking.json".format(out_directory, file_name_base)

        data_items, queries_index = data_utils.get_reader_items_field_text_index(file_name)
        id_field = data_utils.id_field
        field_names = data_utils.field_names
        start = time.time()
        items_to_export = []
        queries_items_meta = []
        queries_items_fields = []

        meta_id = 0

        for lid, item in enumerate(data_items):
            items_to_export.append({id_field: item[id_field], "ext_fact_global_ids": []})
            queries_items_fields.append([item[field_names[0]], item[field_names[1]]])

            meta = {"lid": lid, "obj": item}
            queries_items_meta.append(meta)

            if limit_for_debug > 0 and len(queries_items_meta) > limit_for_debug:
                logging.info("Debug break up to {0}".format(limit_for_debug))
                break

        logging.info("%s queries_items pre-processed in %s" % (len(queries_items_fields), time.time() - start))

        logging.info("#" * 10)
        logging.info("Training TfidfVectorizer")
        logging.info("#" * 10)

        start = time.time()
        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=STOP_WORDS, decode_error='replace',
                                tokenizer=LemmaTokenizer())
        queries_index_transformed = tfidf.fit_transform(queries_index)
        logging.info("%s fit in %s s" % (len(queries_index), time.time() - start))

        start = time.time()
        # logging.info("Selects feature pairs..")
        # queries_items_fields_vecs = [queries_index_transformed[feats_ids] for feats_ids in queries_items_fields]

        # print(tfidf.__dict__)
        tfidf_stopwords = tfidf.get_stop_words()

        logging.info("Done in %s s" % (time.time() - start))

        start = time.time()
        logging.info("Transforming %s paragraphs.." % len(paragraphs))
        para_features = tfidf.transform(paragraphs)
        logging.info("%s fit_transform in %s s" % (len(paragraphs), time.time() - start))

        start = time.time()

        res_dists = []
        proc_list = []
        add_label = []

        max_memory = 10 * 1024 * 1024 * 1024  # 10GB
        precompute_sim_matrix = False
        if para_features.shape[0] * queries_index_transformed.shape[0] * 4 > max_memory:
            precompute_sim_matrix = False
            logging.info("The required memory is more than %s MB. Not using similarity matrix..." % str(
                float(max_memory) / 1024))
        else:
            precompute_sim_matrix = True
            logging.info("The required memory is less than %s MB. Not using similarity matrix..." % str(
                float(max_memory) / 1024))
            logging.info(
                "Building similarity matrix for {0} queries to {1}".format(str(queries_index_transformed.shape),
                                                                           str(para_features.shape)))
            logging.info("This will take some time....")
            similarity_matrix = pairwise_distances(queries_index_transformed, para_features, "cosine")

        logging.info("Done in {0} s!".format(time.time() - start))

        start_single = time.time()

        logging.info("Items to process:%s" % len(queries_items_fields))
        for i, item_feats in enumerate(queries_items_fields):
            if precompute_sim_matrix:
                feat_similarities = similarity_matrix[item_feats]
            else:
                feat_similarities = pairwise_distances(queries_index_transformed[item_feats], para_features, "cosine")

            dists = combine_similarities(feat_similarities, max_facts_per_choice,
                                         combine_feat_scores=combine_feat_scores)

            res_dists.append(dists)
            if (i + 1) % 100 == 0:
                logging.info("Item %s in %s s" % (i + 1, time.time() - start_single))

        logging.info("Retrieve results for  in %s in %s" % (len(paragraphs), time.time() - start))

        out_file_ranks = codecs.open(data_knowledge_ranking_file_name, mode="wb", encoding="utf-8")
        lid_to_proc_id = {item["lid"]: i for i, item in enumerate(queries_items_meta)}
        for i, item in enumerate(items_to_export):
            if i in lid_to_proc_id:
                item["ext_fact_global_ids"] = res_dists[lid_to_proc_id[i]]
            out_file_ranks.write(json.dumps(item))
            out_file_ranks.write("\n")
        logging.info("Rankings saved to file %s" % data_knowledge_ranking_file_name)

        item_id = 0
        for item_id in range(5):
            debug_item = queries_items_meta[item_id]

            def print_item_info(item_obj):
                logging.info("Item obj:")
                logging.info(item_obj)
                logging.info("Processed")
                logging.info(queries_items_fields[item_id])
                # logging.info("Type:%s" % queries_item_obj["obj"]["Type"])
                #         # logging.info("Sense:%s" % queries_item_obj["obj"]["Sense"])
                logging.info("Top %s facts:" % max_facts_per_choice)
                for fid, fact in enumerate(res_dists[item_id]):
                    logging.info("%s - %s - %s " % (fid, fact[1], paragraphs[fact[0]]))

            print_item_info(debug_item)

    # save settings to file
    proc_settings = {
        "data_run_name_dir": out_directory,
        "knowledge_file_json": knowledge_file_json,
        "know_run_name": know_run_name,
        "max_facts_per_item": max_facts_per_choice,
    }

    with open(os.path.join(out_directory, know_run_name + ".proc_settings.json"), mode="w") as f_proc:
        f_proc.write(json.dumps(proc_settings))
