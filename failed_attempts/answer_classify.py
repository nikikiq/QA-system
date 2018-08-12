import nltk
import json
from tqdm import tqdm
from ner_test07 import parse_docs
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sent_retrieval import eval_query, process_query, prepare_doc, get_word_sets
from ranking import add_answer_properties, get_question_type, get_open_class_words
from word2vec import get_word2vec_model
import numpy as np
import random

REGRESSOR_TYPE = LinearRegression

# w2v = get_word2vec_model()

def get_pos_tag(answer, doc_set):
	tagged = nltk.pos_tag(nltk.word_tokenize(doc_set[answer['id']]))

def make_feature_vector(answer, question_words, question_type, open_class_words, doc_set, sentences):
	added = add_answer_properties(question_words, question_type, open_class_words, answer, doc_set, sentences)
	features = []
	# open_class_words_sum = None
	# for word in open_class_words:
	# 	if word in w2v.wv:
	# 		if open_class_words_sum == None:
	# 			open_class_words_sum = w2v.wv[word]
	# 		else:
	# 			open_class_words_sum += w2v.wv[word]
	# features.extend(open_class_words_sum)

	# Answer type match
	features.append(float(added['matches_question_type']))

	# Pattern match (TODO)

	# Number of matched question keywords
	features.append(float(len(set(question_words) & set(doc_set[answer['id']]))))

	# Keyword distance
	dist = added['dist_to_open_words']

	# Novelty factor
	features.append(float(added['appear_in_question']))

	# Apposition features (TODO)

	# Punctuation location
	punct = False
	print answer
	left = answer['start_pos'] - 1
	right = answer['end_pos'] + 1
	print left
	print right
	if left >= 0:
		if answer['pos_sent'][left][0] in ',.!?;':
			punct = True
	if right < len(answer['pos_sent']):
		if answer['pos_sent'][right][0] in ',.!?;':
			punct = True
	features.append(float(punct))

	# Sequences of question terms (TODO)
	# qwords = set(question_words)
	# answer_qwords = [e[1] for e in answer['pos_sent'] if e[1] in question_words]
	# i = 0
	# seq = 0
	# longest_seq = 0
	# for w in answer_qwords:
	# 	if w == qwords:
	# 		seq += 1
	# 	i += 1

	rank = added['sent_retrieval_rank']
	features.append(1000.0 if rank == None else float(rank))

	
	features.append(1000.0 if dist == None else float(dist))
	# TODO add more...
	return np.array(features)

def train_regressor(sample_trial_size=None, sample_qa_size=None):
	print "start training answer regressor"
	print sample_trial_size
	print sample_qa_size


	# load json
	with open('QA_train.json') as file:
		dataset = json.load(file)
	
	# take random sample if specified
	if sample_trial_size:
		indices = random.sample(xrange(len(dataset)), sample_trial_size)
		dataset = [dataset[i] for i in indices]

	# get all usable (NER successful) training data
	correct_cases = []
	incorrect_cases = []

	for trial in tqdm(dataset):
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		no_docs = len(doc_set)
		word_sets = get_word_sets(doc_set)
		entities = parse_docs(trial['sentences'])
		correct_entities = []

		qa_list = trial['qa']
		# take random sample if specified
		if sample_qa_size:
			indices = random.sample(xrange(len(qa_list)), sample_qa_size)
			qa_list = [qa_list[i] for i in indices]

		for qa in tqdm(qa_list):
			# (lazy) sentence retrieval
			query = process_query(qa['question'])
			question_words = [w.lower() for w in qa['question']]
			question_type = get_question_type(question_words)
			open_class_words = get_open_class_words(question_words)
			possible_sents = eval_query(query, posting, word_sets, no_docs)
			num_incorrect = 0
			for e in entities:
				if e['answer'] == qa['answer'] and e['id'] == qa['answer_sentence']:
					correct_cases.append(make_feature_vector(
						e,
						question_words,
						question_type,
						open_class_words,
						doc_set,
						possible_sents
					))
				else:
					if num_incorrect < 5:
						incorrect_cases.append(make_feature_vector(
							e,
							question_words,
							question_type,
							open_class_words,
							doc_set,
							possible_sents
						))
						num_incorrect += 1

	features = np.vstack(( np.array(correct_cases), np.array(incorrect_cases) ))
	outcomes = np.hstack(( np.full((len(correct_cases),), 1.0), np.full((len(incorrect_cases),), 0.0) ))
	regr = REGRESSOR_TYPE()
	regr.fit(features, outcomes)
	return regr

def filter_answers(regr, answers, question_words, question_type, open_class_words, doc_set, sentences):
	for answer in answers:
		answer['score'] = regr.predict(make_feature_vector(answer, question_words, question_type, open_class_words, doc_set, sentences))[0]
	return sorted(answers, key=lambda x: x['score'])
	# return [a for a in answers
	# 	if regr.predict(make_feature_vector(a, question_words, question_type, open_class_words, doc_set, sentences)) == 'y']

if __name__ == "__main__":
	train_classifier(sample_trial_size=10, sample_qa_size=2)