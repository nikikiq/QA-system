##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: basic_ner.py
## Description: Basic answer extraction module.
##
## Team: Mainframe
## Members:
## Name         | Student ID
## Kuan QIAN    | 686464
## Zequn MA     | 696586
## Yueni CHANG  | 884622
##

import os
import nltk
import json
from nltk.tag import StanfordNERTagger


os.environ['CLASSPATH'] = '/usr/local/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/local/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'


classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
st = StanfordNERTagger(classifier,jar)
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def process_doc_ner(doc_set):
    # doc as a single sentence
    new_docs = []
    for doc in doc_set:
        doc = word_tokenizer.tokenize(doc)
        new_docs.append(doc)
    return new_docs 

def get_next_tag (tagged_sent,cur_tag):
    for word, tag,i in tagged_sent:
        if tag != cur_tag:
            pos = tagged_sent.index((word,tag,i))
            return pos

def get_continuous_chunks(tagged_sents):
    sents_chunks = []
    for tagged_sent in tagged_sents:
        tagged_sent.append(('end','END','END'))
        continuous_chunk = []
        sent_not_empty = True


        while sent_not_empty:
            cur_tag = tagged_sent[0][1]
            pos = get_next_tag(tagged_sent,cur_tag)
            chunk = tagged_sent[0:pos]
            continuous_chunk.append(chunk)
            tagged_sent = tagged_sent[pos:]

            if len(tagged_sent) == 1:
                sent_not_empty = False

        sents_chunks.append(continuous_chunk)
    return sents_chunks

def get_tagged(processed_docs):
    ner_tagged_sents = st.tag_sents(processed_docs)
    tagged_sents = []

    
    for sent in ner_tagged_sents:
        tagged_sent =[]
        no_tokens = len(sent)

        for i in range (0,no_tokens):
            token = sent[i][0]
            tag = sent[i][1]
            if token != '':
                if tag != 'O':
                      tagged_sent.append((token,tag,i))

                else:
                    if i != 0 and token[0].isupper():
                        tag = 'OTHER'
                        tagged_sent.append((token,tag,i))

                    elif token.isdigit():
                        tag = 'NUMBER'
                        tagged_sent.append((token,tag,i))
                    else:
                        tagged_sent.append((token,tag,i))

        tagged_sents.append(tagged_sent)
    return tagged_sents


def parse_docs(doc_set):
    processed_docs = process_doc_ner(doc_set)
    no_docs = len(processed_docs)
    tagged_sents = get_tagged(processed_docs)

    name_entity_list = get_continuous_chunks(tagged_sents)

    doc_ne_pairs = []
    for i in range (0,no_docs):
        name_entity = name_entity_list[i]
        # name_entity_str = [" ".join([token for token, tag in ne]) for ne in name_entity]
        name_entity_pairs = [(i," ".join([token for token, tag, pos in ne]), ne[0][1],ne[0][2],ne[-1][2]) for ne in name_entity]
        for sent_id,entity,tag,start_i,end_i in name_entity_pairs:
            if tag != '0':
                if tag == 'ORGANIZATION':
                    doc_ne_pairs.append((sent_id,entity,'OTHER',start_i, end_i))
                else:
                    doc_ne_pairs.append((sent_id,entity, tag, start_i, end_i))
    return doc_ne_pairs
