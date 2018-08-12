from nltk.parse.stanford import StanfordParser

import os

import nltk
from nltk.tree import ParentedTree

os.environ['CLASSPATH'] = '$CLASSPATH:/usr/share/stanford-parser/stanford-parser.jar:/usr/share/stanford-parser/stanford-parser-3.7.0-models.jar'
# os.environ['STANFORD_PARSER'] = "stanford-parser.jar"
# os.environ['STANFORD_MODELS'] = "$STANFORD_MODELS:stanford-parser-3.7.0.models.jar"

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def get_path_len(path1, path2):
    while path1[0] == path2[0]:
        path1 = path1[1:]
        path2 = path2[1:]
    return len(path1) + len(path2)

def get_tree_dist(tree, value1, value2):
    paths = tree.treepositions(order='preorder')
    path1 = None
    path2 = None
    for p in paths:
        if tree[p] == value1:
            path1 = p
        elif tree[p] == value2:
            path2 = p
        if path1 and path2:
            break
    if not (path1 and path2):
        return None
    return get_path_len(path1, path2)


def cfg_path_dist(sentence, part_a, part_b):
    return cfg_path_dist_tagged(nltk.pos_tag(nltk.word_tokenize(sentence)), nltk.word_tokenize(part_a), nltk.word_tokenize(part_b))

def cfg_path_dist_tagged(sentence_tagged, a_words, b_words):
    try:
        tree = next(parser.tagged_parse(sentence_tagged))
    except ValueError:
        return None

    dists = []
    for a in a_words:
        for b in b_words:
            try:
                dists.append(get_tree_dist(tree, a, b))
            except ValueError:
                pass
    return min(dists) if dists else None
