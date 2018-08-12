##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: wordnet_func.py
## Description: Collection of functions using wordnet interface from nltk.
##
## Team: Mainframe
## Members:
## Name         | Student ID
## Kuan QIAN    | 686464
## Zequn MA     | 696586
## Yueni CHANG  | 884622
##

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import ranking

stopwords_set = set([w.lower() for w in stopwords.words()])

# get senses of a word and their frequencies, sorted by freqency descending
def get_sense_freqs(word, pos=None):
    sense_freqs = []
    for s in wn.synsets(word):
        if pos:
            if pos != s.pos():
                continue
        freq = 0
        for lemma in s.lemmas():
            if lemma.name() == word:
                freq += lemma.count()
        sense_freqs.append( (s, freq) )
    return sorted(sense_freqs, key=lambda x: x[1], reverse=True)

def get_pri_sense(word, pos=None):
    sense_freqs = get_sense_freqs(word)
    return None if len(sense_freqs) == 0 else sense_freqs[0][0]

def get_head_word(question_words):
    what_idx = None
    which_idx = None
    wh_idx = None
    if 'what' in question_words:
        what_idx = question_words.index('what')
    if 'which' in question_words:
        which_idx = question_words.index('which')
    if what_idx == None and which_idx == None:
        return None
    if what_idx == None:
        wh_idx = which_idx
    elif which_idx == None:
        wh_idx = what_idx
    else:
        wh_idx = min(what_idx, which_idx)
    for w in pos_tag(question_words, tagset='universal')[wh_idx+1:]:
        if w[0] not in stopwords_set and w[1] == 'NOUN':
            return w[0]
    return None

def get_head_word_synset(question_words):
    w = get_head_word(question_words)
    return get_pri_sense(w, pos=wn.NOUN)

def get_max_wup_similarity(answer, target):
    sims = [max( [ s.wup_similarity(target) for s in wn.synsets(w) ] )
        for w in word_tokenize(answer) ]
    print sims
    return max(sims)

def get_all_hyponyms(synset):
    hyponyms = synset.hyponyms()
    collected = set(hyponyms)
    for h in synset.hyponyms():
        collected |= get_all_hyponyms(h)
    return collected
