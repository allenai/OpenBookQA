#!/bin/bash

set -e
set -x

pip install -r requirements.txt

# Temporary fix to the build whilst NLTK sort stuff out. TODO(Mark): revert this.
python -m nltk.downloader -u https://pastebin.com/raw/D3TBY4Mj punkt

python -m spacy download en

python -m nltk.downloader wordnet

