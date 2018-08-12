import os
import nltk
import json
from nltk.tag import StanfordNERTagger
import Tkinter
from nltk.corpus import stopwords
from random import randint


#nltk.download('words')

os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'


classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
STOPWORDS = set(stopwords.words('english'))
name_list = nltk.corpus.names
names = set([name for file in ('female.txt','male.txt') for name in name_list.words(file)])

text_file = open("units.txt", "r")
units = text_file.read().split("', '")
text_file.close()

units = set(units)

# sent1 = 'The NASCAR Sprint Cup Series holds two exhibition events annually - the Sprint Unlimited, held at Daytona International Speedway at the start of the season, and the NASCAR Sprint All-Star Race, held at Charlotte Motor Speedway midway through the season.'
# sent2 = 'J.K. Rowling, the English writer, lives in the South of Boston Common and she has 13,231 books. '
# #s1 = word_tokenizer.tokenize(sent)
# s1 = nltk.word_tokenize(sent1)
# s2 = nltk.word_tokenize(sent2)

# # ner = st.tag(s2)
# pos1 = nltk.pos_tag(s1)
# pos2 = nltk.pos_tag(s2)
# test = nltk.ne_chunk(pos)

def process_doc_ner(doc_set):
    # doc as a single sentence
    new_docs = []
    # new_docs_pu = []
    for doc in doc_set:
        # doc_pu = nltk.word_tokenize(doc)
        doc = nltk.word_tokenize(doc)
        new_docs.append(doc)
        # new_docs_pu.append(doc_pu)
    return new_docs

def get_next_tag (tagged_sent,cur_tag):
    for token, tag, i in tagged_sent:
        if tag != cur_tag:
            pos = tagged_sent.index((token,tag,i))
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

def ner_tagger(processed_docs):
    ner_tagged_sents = st.tag_sents(processed_docs)
    tagged_sents = []
    for ner_sent in ner_tagged_sents:
    	no_tokens = len(ner_sent)
    	tagged_sent = []

    	for j in range (0,no_tokens):
    		token = ner_sent[j][0]
    		tag = ner_sent[j][1]

    		tagged_sent.append((token,tag,j))

    	tagged_sents.append(tagged_sent)

    return tagged_sents


def subfinder(sent, entity):
    matches = []
    tokens = nltk.word_tokenize(entity)
    pl = len(tokens)
    n = len(sent)
    
    if pl != 0:
	    for i in range(0,n):
	        if sent[i] == tokens[0] and i+pl <= n and sent[i:i+pl] == tokens:
	        	matches.append((i,i+pl-1))
    return matches


def get_np(processed_docs,pos_tagged_sents,no_docs):
	cp = nltk.RegexpParser('''
		NP: {<DT>?<JJ>*<NN>}
		NP: {<DT>?(<NNP>|<NNPS>)+<IN>?(<NNP>|<NNPS>)+}
		NP: {<DT>?<NNP>+}
		NP: {<CD>+}''')
# <DT>

	trees = cp.parse_sents(pos_tagged_sents)

	entities = []

	for tree in trees:
		for subtree in tree.subtrees():
			if subtree.label() == 'NP':
				et = subtree.leaves()
				entity = [token for token, tag in et]
				entities.append(entity)


	ner_pairs = []

	ner_entities = st.tag_sents(entities)
	for ner_entity in ner_entities:
		no_tokens = len(ner_entity)
		entity = " ".join([token for token, tag in ner_entity])

		for j in range (0,no_tokens):
			token = ner_entity[j][0]
			tag = ner_entity[j][1]
			if token != '':

				if tag != 'O':
					if tag == 'ORGANIZATION':
						ner_pairs.append((entity,'OTHER'))
					else:
						ner_pairs.append((entity,tag))
					break
				else:
					if nltk.pos_tag([token])[0][1] == 'CD':
						ner_pairs.append((entity,'NUMBER'))
						break

					if j == no_tokens - 1:
						ner_pairs.append((entity,'OTHER'))

	ner_pairs = set(ner_pairs)
	answers= []
	for entity, tag in ner_pairs:
		for i in range (0,no_docs):
			pos_sent = pos_tagged_sents[i]
			sent = processed_docs[i]
			matches = subfinder(sent,entity)
			for match in matches:
				answers.append({'id':i, 'answer':entity, 'type':tag, 'start_pos':match[0],'end_pos':match[1]})


	return answers

def parse_docs(doc_set):
    processed_docs = process_doc_ner(doc_set)
    pos_tagged_sents = nltk.pos_tag_sents(processed_docs)
    no_docs = len(processed_docs)

    #ner
    tagged_sents = ner_tagger (processed_docs)
    name_entity_list = get_continuous_chunks(tagged_sents)
    doc_ne_pairs01 = []
    for i in range (0,no_docs):
        name_entity = name_entity_list[i]
        name_entity_pairs = [(i," ".join([token for token, tag, start in ne]), ne[0][1],ne[0][2],ne[-1][2]) for ne in name_entity]
        for sent_id,entity,tag,start_i,end_i in name_entity_pairs:
            if tag != 'O':
            	if tag == 'ORGANIZATION':
            		doc_ne_pairs01.append({'id':sent_id,'answer':entity,'type':'OTHER','start_pos':start_i,'end_pos':end_i})
            	else:
                	doc_ne_pairs01.append({'id':sent_id,'answer':entity,'type':tag,'start_pos':start_i,'end_pos':end_i})

    # np
    doc_ne_pairs02 = get_np(processed_docs,pos_tagged_sents,no_docs)
    answers = doc_ne_pairs01 + doc_ne_pairs02



    return answers






if __name__ == '__main__':
    with open('QA_dev.json') as dev_file:
        devs = json.load(dev_file)



        # re = parse_docs(doc_set)
        # for r in re:
        #     print r['pos_sent']
    total = 0
    match_dict = 0

    for dev in devs:
    	doc_set = dev['sentences']
    	entities_dict = parse_docs(doc_set)
    	print entities_dict
    	answers = dev['qa']
    	total_d = len(answers)
    	total = total + total_d

    	for answer in answers:

	    	answer_id = answer['answer_sentence']
	    	question = answer['question']
	    	a = answer['answer']


	    	for entity in entities_dict:
	    		if entity['id'] == answer_id and entity['answer'] == a:
	    			match_dict = match_dict + 1
	    			print question,
	    			print ''
	    			print entity['type'],a
	    			print ' '
	    			break

	print match_dict
	print total
	print float(match_dict)/float(total)

# print match
# print total
# print float(match)/float(total)

# for re in result:
# 	t =  next(re)
# 	print t.flatten()

# print pos

# def process_doc_ner(doc_set):
#     # doc as a single sentence
#     new_docs = []
#     for doc in doc_set:
#         doc = word_tokenizer.tokenize(doc)
#         new_docs.append(doc)
#     return new_docs 

# with open('QA_dev.json') as dev_file:
#     dev = json.load(dev_file)
        
# for i in range (0,5):
#     doc_set = dev[i]['sentences']
#     no_docs = len(doc_set)
#     processed_docs = process_doc_ner(doc_set)
#     ner_tagged_sents = st.tag_sents(processed_docs)
#     pos_tagged_sents = nltk.pos_tag_sents(processed_docs)

#     for i in range (0, no_docs):
#     	print ner_tagged_sents[i][0],pos_tagged_sents[i][0]
#       for sent_id,entity_list in entities:
#           if sent_id == answer_id:
#               if a in entity_list:
#                   match = match + 1
#                   # print question,
#                   # print entity_list,a,sent_id
#                   # print ' '
#                   # for entity in entities_dict:
#                   #   if entity['id'] == answer_id:
#                   #       print entity 