##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: ranking.py
## Description: Module used for answer ranking.
##
## Team: Mainframe
## Members:
## Name         | Student ID
## Kuan QIAN    | 686464
## Zequn MA     | 696586
## Yueni CHANG  | 884622
##

from qtype_classifier import get_classifier, lemmatize_doc, get_que_bow
from wordnet_func import get_head_word
from sent_retrieval import remove_stop
from functools import cmp_to_key
from tqdm import tqdm

import nltk

word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
vectorizer, classifier = get_classifier()

def get_question_type(question_words):
	"""Determine question type.

	Args:
		question_words ([str]): list of words in question

	Returns:
		(str): type of question as a string
	"""
	q_bow = {}
	for word in question_words:
		q_bow[word] = q_bow.get(word, 0) + 1
	q_vec = vectorizer.transform(q_bow)
	q_type = classifier.predict(q_vec)
	return q_type[0]

# Return True if not items contains all elements.
def contains_all(items, elems):
	for e in elems:
		if e not in items:
			return False
	return True

# Get all open class words from a question.
def get_open_class_words(question_words):
	tagged = nltk.pos_tag(question_words, tagset="universal")
	# consider pronouns, determiners, conjunctions, and prepositions as closed class
	return remove_stop([p[0] for p in tagged if p[1] in \
			["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"] \
			and p[0] not in ['what', 'which', 'how', 'who', 'where']])

def get_dist_to_question_word(target_words, sentence_words, entity):
	# get positions of question words
	question_words_pos = []
	for w in target_words:
		for i in range(len(sentence_words)):
			if w == sentence_words[i]:
				question_words_pos.append(i)
	# cannot proceed if no such closed word in sentence
	if len(question_words_pos) == 0:
		return None
	answer_start_pos = entity['start_pos']
	answer_end_pos = entity['end_pos']
	# calculate distance and take sum
	dists = [ min(abs(p-answer_start_pos), abs(p-answer_end_pos))
		for p in question_words_pos ]
	return sum(dists) / float(len(dists))

def cmp_answer(a, b):
	# First, answers whose content words all appear
	# in the question should be ranked lowest.
	if a['appear_in_question'] != b['appear_in_question']:
		return b['appear_in_question'] - a['appear_in_question']
	# Second, answers which match the question type
	# should be ranked higher than those that don't;
	if a['matches_question_type'] != b['matches_question_type']:
		return a['matches_question_type'] - b['matches_question_type']
	# consider relavance (rank) in sentence retrieval
	if a['sent_retrieval_rank'] != b['sent_retrieval_rank']:
		return b['sent_retrieval_rank'] - a['sent_retrieval_rank']
	# Third, among entities of the same type, the
	# prefered entity should be the one which is closer
	# in the sentence to a closed-class word from the question.
	if a['dist_to_open_words'] != b['dist_to_open_words']:
		if a['dist_to_open_words'] == None:
			return -1
		elif b['dist_to_open_words'] == None:
			return 1
		else:
			return b['dist_to_open_words'] - a['dist_to_open_words']
	return 0

# Generated an answer from specific information.
def add_answer_properties(question_words, question_type, open_class_words, answer, doc_set, sentences):
	answer_words = word_tokenizer.tokenize(answer['answer'].lower())
	answer_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[answer['id']]) ]
	added = dict(answer)
	added['sent_retrieval_rank'] = sentences.index(answer['id']) if answer['id'] in sentences else None
	added['appear_in_question'] = contains_all(question_words, answer_words)
	added['matches_question_type'] = answer['type'] == question_type
	added['dist_to_open_words'] = get_dist_to_question_word(open_class_words, answer_sent_words, answer)
	return added

def get_best_answer(question, answers, doc_set, sentences):
	"""Return the best answer from answers to a question.

	Args:
		answers [(str, str, str)]: a list of answers,
			each being a 3-tuple of (sentence, entity, entity type)
		doc_set [str]: a list of all answers, indexed by answer id
		sentences [int]: a list of sentence id's sorted by relavance

	Returns:
		(str, str, str): the best answer to the question
	"""
	question_words = [ w.lower() for w in word_tokenizer.tokenize(question) ]
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in answers
	]
	key_func = cmp_to_key(cmp_answer)
	return max(answers_added, key=key_func)

# Returns a list of answers ranked on top.
def get_top_answers(question, answers, doc_set, sentences):
	question_words = [ w.lower() for w in word_tokenizer.tokenize(question) ]
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in tqdm(answers)
	]
	key_func = cmp_to_key(cmp_answer)
	top = sorted(answers_added, reverse=True, key=key_func)
	return top
