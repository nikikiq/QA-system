##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: sent_retrieval.py
## Description: Sentence retrieval module of QA system.
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

import nltk

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

# Remove stopwords in a tokenized sentence.
def remove_stop(sentence):
	new = []
	for word in sentence:
		if not word in STOPWORDS:
			new.append(word)
	return new

# Lemmatize a word.
def lemmatize(word):
	lemma = lemmatizer.lemmatize(word,'v')
	if lemma == word:
		lemma = lemmatizer.lemmatize(word,'n')
	return lemma

# Lemmatize a sentence.
def lemmatize_doc(document):
	output = []
	for word in document:
		if word.isalnum():
			output.append(lemmatize(word.lower()))
	return output

# Get term frequency and document frequency information from a
# set of document.
def get_freqencies(documents):
	term_freqs = []
	doc_freq = {}
	for document in documents:
		document = word_tokenizer.tokenize(document)
		document = lemmatize_doc(document)
		document = remove_stop(document)
		term_freq = {}
		for word in document:
			term_freq[word] = term_freq.get(word, 0) + 1
		for word in list(set(document)):
			doc_freq[word] = doc_freq.get(word, 0) + 1
		term_freqs.append(term_freq)
	return term_freqs, doc_freq

# Get tf*idf information.
def get_tf_idf(term_freqs, doc_freq):
	tf_idf = []
	words = doc_freq.keys()
	for doc_dict in term_freqs:
		doc_vector = []
		words_in_doc = set(doc_dict.keys())
		for word in words:
			if word in words_in_doc:
				doc_vector.append(doc_dict[word] * 1.0 / doc_freq[word])
			else: doc_vector.append(0)
		tf_idf.append(doc_vector)
	return words, tf_idf

# Make posting lists.
def get_inverted_index(words, tf_idf):
	posting = {}
	for i in range(len(words)):
		posting[words[i]] = posting.get(words[i], [])
		for doc_id in range(len(tf_idf)):
			word_weight = tf_idf[doc_id][i]
			if word_weight != 0:
				posting[words[i]].append((doc_id, word_weight))
	return posting

# Preprocess a question to query which can then be evaluated.
def process_query(query):
	return remove_stop(lemmatize_doc(word_tokenizer.tokenize(query)))

# Generate posting lists from a set of document.
def prepare_doc(doc_set):
	term_freqs, doc_freq = get_freqencies(doc_set)
	words, tf_idf = get_tf_idf(term_freqs, doc_freq)
	posting = get_inverted_index(words, tf_idf)
	return posting

# Turn sentences into sets of word.
def get_word_sets(doc_set):
	word_sets = []
	for doc in doc_set:
		word_sets.append(set(remove_stop(lemmatize_doc(word_tokenizer.tokenize(doc)))))
	return word_sets

# Evaluate a query with posting list and word sets to return most relevent sentence.
def eval_query(query, posting, word_sets, n=1):
	scores = {}
	for term in query:
		posting_list = posting.get(term, [])
		for (doc_id, weight) in posting_list:
			# smooth with the proportion of words in query which overlap
			# with target sentence.
			scores[doc_id] = scores.get(doc_id, 0) + \
								weight \
								* len(set(query).intersection(word_sets[doc_id])) \
								/ len(set(query))
	sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
	return [d for d, w in sorted_scores][:n]

# Retrieve most relevant sentence for a question.
def retrieve_sentences(question, doc_set, n=1):
	query = process_query(question)
	posting = prepare_doc(doc_set)
	word_sets = get_word_sets(doc_set)
	return eval_query(query, posting, word_sets, n)
