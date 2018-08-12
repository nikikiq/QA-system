from nltk.parse.stanford import StanfordParser
import nltk.parse
import nltk
import os
from nltk.tree import ParentedTree
import Tkinter
import json
from collections import defaultdict
import random

os.environ['CLASSPATH'] = '$CLASSPATH:/usr/share/stanford-parser/stanford-parser.jar:/usr/share/stanford-parser/stanford-parser-3.7.0-models.jar'
parser = StanfordParser(model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


with open('QA_train.json') as dev_file:
    dev = json.load(dev_file)


answers = []
for trial in dev:
    for question in trial['qa']:
        answer = question['answer']
        answer = nltk.pos_tag(nltk.word_tokenize(answer))
        answers.append(answer)

stat = defaultdict(dict)

for j in range (0,70159):
	answer = answers[j]
	for token,tag in answer:
		if token == "(":
			stat[j]['l'] = stat[j].get('l',0) + 1
		elif token == ")":
			stat[j]['r'] = stat[j].get('r',0) + 1
nm = []
for item in stat.items():

	if item[1].get('l',0) != item[1].get('r',0):
		nm.append(item[0])

pure_answers = []
for i in range (0,70159):
	if i not in nm:
		pure_answers.append(answers[i])

sample = random.sample(pure_answers,50)
print sample
trees = parser.tagged_parse_sents(pure_answers[0:100])


for tree in trees:
	t =  next(tree)
	# t.draw()
	print t.leaves()


