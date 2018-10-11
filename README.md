# OpenBookQA Models

This repository provides code for various baseline models reported in the EMNLP-2018 paper
introducing the OpenBookQA dataset:
[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://www.semanticscholar.org/paper/24c8adb9895b581c441b97e97d33227730ebfdab)

```bib
@inproceedings{OpenBookQA2018,
 title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
 author={Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
 booktitle={EMNLP},
 year={2018}
}
```

Please visit the [OpenBookQA Leaderboard](https://leaderboard.allenai.org/open_book_qa) for the latest on this challenge!


# Setting Up the Environment

1. Create the `obqa` environment using Anaconda

  ```
  conda create -n obqa python=3.6
  ```

2. Activate the environment

  ```
  source activate obqa
  ```

3. Install the requirements in the environment:

  ```
  bash scripts/install_requirements.sh
  ```

4. Install pytorch

  Mac (no CUDA): `conda install pytorch==0.4.0 torchvision==0.2.1 -c pytorch`
  Linux (no CUDA): `conda install pytorch-cpu==0.4.0 torchvision-cpu==0.2.1 -c pytorch`

  Linux (with CUDA 8): `conda install pytorch==0.4.0 torchvision==0.2.1 cuda80 -c pytorch`
  Linux (with CUDA 9.0): `conda install pytorch==0.4.0 torchvision==0.2.1 -c pytorch`


# Downloading and Preparing Data

Download the OpenBookQA dataset and embeddings using the script below.
Note that this includes downloading `glove.840B.300d.txt.gz`, a 2GB file
containing 300-dimensional [GloVe word embeddings](https://nlp.stanford.edu/projects/glove)
trained on 840B tokens, which can take several minutes.
If you already have this file, you might consider altering the script.

 ```
 bash scripts/download_and_prepare_data.sh
 ```


# Training/Evaluating Neural Baselines for OpenBookQA

If you use the script below, you might want to first look at
``scripts/experiments/qa/run_experiment_openbookqa.sh`` and set the
``EXPERIMENTS_OUTPUT_DIR_BASE`` environment variable to a directory
where you want to save the output of the experiments.
Default is ``_experiments``.

Note: If you want to use GPU for the experiments, make sure to change the
`trainer.cuda_device` setting to the desired CUDA device id. Default is `-1` (no GPU).
You can also use `scripts/experiments/qa/run_experiment_openbookqa_gpu.sh` (automatically sets `trainer.cuda_device` to CUDA device `0`)
instead of `scripts/experiments/qa/run_experiment_openbookqa.sh` in the experiments commands below.

## 1. Without External Knowledge

### 1.1 Question-to-Choice Model

Experiments with pre-trained [GloVe](https://nlp.stanford.edu/projects/glove) embedding vectors:
```
config_file=training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

Experiments with [ELMo](https://allennlp.org/elmo) contextual word representations:
```
config_file=training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice_elmo.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

### 1.2 ESIM Model

Experiments with Glove pre-trained embeddings vectors:
```
config_file=training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

Experiments with ELMo contextual representations:
```
config_file=training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim_elmo.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

## 2. Knowledge-Enhanced Models

### 2.1. Retrieve external knowledge

#### 2.1.1. Open Book Knowledge (1326 Science facts)

Rank OpenBook (Science) knowledge facts for the given question:
```
DATA_DIR_ROOT=data/
KNOWLEDGE_DIR_ROOT=data/knowledge
OPENBOOKQA_DIR=${DATA_DIR_ROOT}/OpenBookQA-V1-Sep2018

ranking_out_dir=${OPENBOOKQA_DIR}/Data/Main/ranked_knowledge/openbook
mkdir -p ${ranking_out_dir}
data_file=${OPENBOOKQA_DIR}/Data/Main/full.jsonl
know_file=${KNOWLEDGE_DIR_ROOT}/openbook.csv

PYTHONPATH=. python obqa/data/retrieval/knowledge/rank_knowledge_for_mc_qa.py \
                     -o ${ranking_out_dir} -i ${data_file} \
                     -k ${know_file} -n tfidf --max_facts_per_choice 100 \
                     --limit_items 0
```

#### 2.1.2. Commonsense Knowledge

##### Open Mind Common Sense part of ConceptNet (cn5omcs)

```
DATA_DIR_ROOT=data/
KNOWLEDGE_DIR_ROOT=data/knowledge
OPENBOOKQA_DIR=${DATA_DIR_ROOT}/OpenBookQA-V1-Sep2018

ranking_out_dir=${OPENBOOKQA_DIR}/Data/Main/ranked_knowledge/cn5omcs
mkdir -p ${ranking_out_dir}
data_file=${OPENBOOKQA_DIR}/Data/Main/full.jsonl
know_file=${KNOWLEDGE_DIR_ROOT}/CN5/cn5_omcs.json

PYTHONPATH=. python obqa/data/retrieval/knowledge/rank_knowledge_for_mc_qa.py \
                     -o ${ranking_out_dir} -i ${data_file} \
                     -k ${know_file} -n tfidf --max_facts_per_choice 100 \
                     --limit_items 0
```


##### WordNet part of ConceptNet (cn5wordnet)

```
DATA_DIR_ROOT=data/
KNOWLEDGE_DIR_ROOT=data/knowledge
OPENBOOKQA_DIR=${DATA_DIR_ROOT}/OpenBookQA-V1-Sep2018

ranking_out_dir=${OPENBOOKQA_DIR}/Data/Main/ranked_knowledge/cn5wordnet
mkdir -p ${ranking_out_dir}
data_file=${OPENBOOKQA_DIR}/Data/Main/full.jsonl
know_file=${KNOWLEDGE_DIR_ROOT}/CN5/cn5_wordnet.json

PYTHONPATH=. python obqa/data/retrieval/knowledge/rank_knowledge_for_mc_qa.py \
                     -o ${ranking_out_dir} -i ${data_file} \
                     -k ${know_file} -n tfidf --max_facts_per_choice 100 \
                     --limit_items 0
```

#### 2.1.3. Retrieve "Gold" Fact from the Open Book (Oracle)

Note: This is Oracle knowledge -- a hypothetical setting that *assumes access to
the gold science fact*. The goal here is to allow research effort to focus on
the sub-challenges of retrieving the missing commonsense knowledge, and reasoning
with both facts in order to answer the question. A full model for OpenBookQA should,
of course, not rely on such Oracle knowledge.

```
DATA_DIR_ROOT=data/
KNOWLEDGE_DIR_ROOT=data/knowledge
OPENBOOKQA_DIR=${DATA_DIR_ROOT}/OpenBookQA-V1-Sep2018

ranking_out_dir=${OPENBOOKQA_DIR}/Data/Main/ranked_knowledge/openbook_oracle
mkdir -p ${ranking_out_dir}
data_file=${OPENBOOKQA_DIR}/Data/Main/full.jsonl
know_file=${OPENBOOKQA_DIR}/Data/Additional/full_complete.jsonl

PYTHONPATH=. python obqa/data/retrieval/knowledge/rank_knowledge_for_mc_qa.py \
                    -o ${ranking_out_dir} -i ${data_file} \
                    -k ${know_file} -n tfidf  --max_facts_per_choice 1 \
                    --limit_items 0 \
                    --knowledge_reader reader_gold_facts_arc_mc_qa_2 \
                    --dataset_reader reader_arc_qa_question_choice_facts
```

### 2.2. Train Knowledge-Enhanced Reader With Above Knowledge

Various baselines that adapt and train the
[Knowledge-Enhanced Reader](https://www.semanticscholar.org/paper/21da1c528d055a134f22e0f8a0b4011fe825a5e7)
model from ACL-2018 for the OpenBookQA setting, using various sources of knowledge.

#### 2.2.1. Oracle Setting

* Oracle Open Book fact + Conceptnet OMCS
(referred to as the `f + ConceptNet` Oracle setup in the paper)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5omcs.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

* Oracle Open Book fact + WordNet
 (referred to as the `f + WordNet` Oracle setup in the paper)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5wordnet.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

#### 2.2.2. Normal (Non-Oracle) Setting

Note: These experiments are **not** reported in the main paper! These are additional
baseline models whose Dev and Test scores are listed below for reference.

* Open Mind Common Sense part of ConceptNet only (cn5omcs)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5omcs.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

* WordNet part of ConceptNet only (cn5wordnet)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5wordnet.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

* Open Book + Open Mind Common Sense part of ConceptNet
(Note: this is **not** the Oracle setup from the paper; instead, science facts from
the Open Book are retrieved based on a TF-IDF similarity measure with the question
and answer choices)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5omcs.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```

* Open Book + WordNet part of ConceptNet
(Note: Similar to above, this is **not** the Oracle setup from the paper)

```
config_file=training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5wordnet.json
bash scripts/experiments/qa/run_experiment_openbookqa.sh ${config_file}
```


# Appendix

## A. Experiments with SciTail, using BiLSTM max-out model

If you are also interested in SciTail entailment task
([Khot et. al 2017](https://www.semanticscholar.org/paper/3ce2e40571fe1f4ac4016426c0606df6824bf619)),
here is a simple BiLSTM max-out model that attains an accuracy of
87% and 85% on the Dev and Test sets, resp.
(without extensive hyper-parameter tuning).

### A.1 Download Scitail Dataset

```
bash scripts/download_and_prepare_data_scitail.sh
```

### A.2 Train the Entailment Model

```
python obqa/run.py train \
    -s _experiments/scitail_bilstm_maxout/ \
    training_config/entailment/scitail/stacked_nn_aggregate_custom_bilstm_maxout_scitail.json
```


## B. Experiments with ARC, using Question-to-Choice BiLSTM max-out model

If you are also interested in the [ARC Challenge](http://data.allenai.org/arc/),
our Question-to-Choice BiLSTM max-out model obtains an
accuracy of 33.9% on the Test set (without extensive hyper-parameter tuning).

### Download ARC Dataset

```
bash scripts/download_and_prepare_data_arc.sh
```

### Train the QA Model

```
python obqa/run.py train \
    -s _experiments/qa_multi_question_to_choices/ \
    training_config/qa/multi_choice/arc/reader_qa_multi_choice_max_att_ARC_Chellenge_full.json
```


# Contact

If you have any questions or comments about the code, data, or models, please
contact Todor Mihaylov, Ashish Sabharwal, Tushar Khot, or Peter Clark.

---
