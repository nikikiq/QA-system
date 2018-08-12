##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: basic_qa_system.py
## Description: Basic QA system with testing functions.
##
## Team: Mainframe
## Members:
## Name         | Student ID
## Kuan QIAN    | 686464
## Zequn MA     | 696586
## Yueni CHANG  | 884622
##

from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from nltk.tag import StanfordNERTagger
from sent_retrieval import *
from basic_ner import parse_docs
from ranking import get_best_answer, get_top_answers, get_question_type, get_open_class_words
from evaluation import is_partial_match, reciprocal_rank

import numpy as np
import nltk
import json
import os
import pprint

pp = pprint.PrettyPrinter(indent=4)
STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def answer_to_tuple(answer):
	return tuple([e for e in answer])

from tqdm import tqdm
def test_with_dev():
	# load json
	with open('QA_dev.json') as dev_file:
		dev = json.load(dev_file)

	reciprocal_ranks = []
	total = 0.0
	num_match_sentences = 0.0
	num_match_first_sentence = 0.0
	num_match_entity = 0.0
	num_match_first_sentence_entity = 0.0
	num_match_correct_sentence_entity = 0.0
	num_match_first_correct_sentence_entity = 0.0
	num_ranking_failed = 0.0
	num_correct_answer = 0.0
	num_partial_answer = 0.0
	for trial in tqdm(dev):
		# make posting list
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		word_sets = get_word_sets(doc_set)
		no_docs = len(doc_set)
		# NER for all sentences
		all_entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			query = process_query(question['question'])
			possible_sents = eval_query(query, posting, word_sets)
			total += 1
			if len(possible_sents) == 0:
				continue

			# check sentence retrieval
			sentence_retrieved = question['answer_sentence'] in possible_sents
			sentence_retrieved_as_first = question['answer_sentence'] == possible_sents[0]
			num_match_sentences += sentence_retrieved
			num_match_first_sentence += sentence_retrieved_as_first
			
			# take all sentences into ranking
			matches = [e for e in all_entities if e[0] in set(possible_sents)]
			if len(matches) == 0:
				continue
			
			# search for the correct answer in matches
			correct_entity = None
			for entity in matches:
				if entity[1] == question['answer']:
					correct_entity = entity
					break

			# check NER result
			entity_extracted = bool(correct_entity)
			entity_extracted_in_correct_sent = False
			entity_extracted_in_first_sent = False
			entity_extracted_in_first_correct_sent = False
			
			if correct_entity:
				num_match_entity += 1
				entity_extracted_in_first_sent = correct_entity[0] == possible_sents[0]
				entity_extracted_in_correct_sent = correct_entity[0] == question['answer_sentence']
			
			num_match_correct_sentence_entity += entity_extracted_in_correct_sent
			num_match_first_sentence_entity += entity_extracted_in_first_sent
			retrieval_and_ner_correct = entity_extracted_in_correct_sent and entity_extracted_in_first_sent
			num_match_first_correct_sentence_entity += retrieval_and_ner_correct
			
			# find best answer
			top_answers = get_top_answers(
					question['question'],
					matches,
					doc_set,
					possible_sents)
			best_match = top_answers[0]

			reciprocal_ranks.append(reciprocal_rank(question['answer'], [a[1] for a in top_answers]))

			if best_match[1] == question['answer']:
				# exact match
				num_correct_answer += 1
			elif retrieval_and_ner_correct:
				num_ranking_failed += 1

			if is_partial_match(best_match[1], question['answer']):
				if possible_sents[0] == question['answer_sentence']:
					num_partial_answer += 1

	print "% sentence retrieved:", num_match_sentences / total
	print "% sentence retrieved as first:", num_match_first_sentence / total
	print "% entity identified:", num_match_entity / total
	print "% entity identified in first sentence:", num_match_first_sentence_entity / total
	print "% entity identified in correct sentence:", num_match_correct_sentence_entity / total
	print "% entity identified in first and correct sentence:", num_match_first_correct_sentence_entity / total
	print "% above but ranking failed:", num_ranking_failed / total
	print "% partial matches in correct sentence:", num_partial_answer / total
	print "% correct best answer:", num_correct_answer / total
	print "Mean reciprocal rank:", np.mean(reciprocal_ranks)

def escape_csv(answer):
	return answer.replace('"','').replace(',','-COMMA-')

import csv
def make_csv():
	# load json
	with open('QA_test.json') as dev_file:
		dev = json.load(dev_file)

	csv_file = open('output.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['id', 'answer'])

	for trial in tqdm(dev):
		# make posting list
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		word_sets = get_word_sets(doc_set)
		no_docs = len(doc_set)
		# NER for all sentences
		entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			query = process_query(question['question'])
			possible_sents = eval_query(query, posting, word_sets)
			if len(possible_sents) == 0:
				writer.writerow( [question['id'], ''] )
				continue
			
			# search for entities in possible sents
			matches = []

			# take all sentences into ranking
			for sent in possible_sents:
				matches.extend([e for e in entities if e[0] == sent])

			if len(matches) == 0:
				writer.writerow( [question['id'], ''] )
				continue
			
			# find best answer
			best_match = get_best_answer(
				question['question'],
				matches,
				doc_set,
				possible_sents)

			writer.writerow( [question['id'], escape_csv(best_match[1]).encode('utf-8')] )

	csv_file.close()


if __name__ == '__main__':
	test_with_dev()
	# make_csv()
