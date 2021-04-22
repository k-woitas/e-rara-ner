#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install and import modules
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
import pandas as pd

# load the NER tagger
# See available sentence tagger: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#list-of-pre-trained-sequence-tagger-modelsentence 
tagger = SequenceTagger.load('dbmdz/flair-historic-ner-onb')  # size: 444MB  GPU

# load corpus file
infile = "corpus_bernensia_ger_LOC_SpaCy.csv"
with open(infile, 'r') as f:            
            corpus = pd.read_csv(f, encoding="UTF-8", usecols=[1,3])

corpus['dbmdz-historic-ner-onb'] = ''
for index in corpus.index[1:176]:
    # use splitter to split text into list of sentences
    sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(corpus['clean_text'][index])]   
    tagger.predict(sentences)   # predict tags for sentences
    loc_ents = []
    for s in sentences:
        for token in s.tokens:
            tag = token.get_tag('ner')
            if tag.value in ['S-LOC', 'B-LOC', 'E-LOC', 'I-LOC']:
                loc_ents.append([token.text, tag.value])
                corpus['dbmdz-historic-ner-onb'][index] = loc_ents
    outfile = "corpus_bernensia_ger_LOC_dbmdz-historic-ner-onb.csv"
    with open(outfile, "w") as f:
        corpus.to_csv(f, index=False, columns=['e_rara_id', 'dbmdz-historic-ner-onb'])
