from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from string import digits
from nltk.corpus import stopwords
import functools
import math
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()
porter_stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

# we might want to remove links at some point...
def is_link(text):
	link_indicators = ["http", ".com", ".org", ".net", ".co.uk"]
	for indicator in link_indicators:
		if indicator in text:
			return True
	return False


# Basic cleanup, remove digits and odd / empty signs
def remove_non_words(words):
	good_words = []
	for word in words:
		if not is_link(word):
			remove_digits = str.maketrans('', '', digits)
			remove_signs = str.maketrans('', '', '+*\',.-;:@!"§$%&/()[]{}=?\\_#|~«»–—‘’“”•…')
			clean_token = word.translate(remove_digits).translate(remove_signs)
			if len(clean_token) > 0 and len(clean_token.translate(str.maketrans('', '', ' '))) > 0:
				good_words.append(clean_token)
	return good_words


# stopwords will have a high frequency but are rarely important, so we might want to remove them
def remove_stopwords(tweet):
	lowered_words = [x.lower() for x in tokenizer.tokenize(tweet["Text"])]
	return [x for x in lowered_words if x not in stops]


# Return the number of unique types over the number of tokens
def lexical_diversity(tweets):
	#words_naive = []
	#words_lowered = []
	#words_stemmed = []
	words_lemmas = []
	for tweet in tweets:
		for token in remove_non_words(tokenizer.tokenize(tweet["Text"])):
			#words_naive.append(clean_token)
			#words_lowered.append(clean_token.lower())
			#words_stemmed.append(porter_stemmer.stem(clean_token.lower()))
			words_lemmas.append(wordnet_lemmatizer.lemmatize(token.lower()))

	#naive_res = len(set(words_naive))/len(words_naive)
	#lowered_res = len(set(words_lowered))/len(words_lowered)
	#stemmed_res = len(set(words_stemmed))/len(words_stemmed)

	return len(set(words_lemmas)) / len(words_lemmas)


# See for each word if it is contained in our "swearwords.txt"
# We got the list from "freewebheaders.com":
# https://www.freewebheaders.com/download/files/full-list-of-bad-words_text-file_2018_07_30.zip
def profanity_share(tweets):
	with open("swearwords.txt") as f:
		defined_swear_words = f.readlines()
	defined_swear_words = [x.strip() for x in defined_swear_words]

	words = []
	swear_words = []
	for tweet in tweets:
		for token in remove_non_words(tokenizer.tokenize(tweet["Text"])):
			words.append(token)
			if token in defined_swear_words:
				swear_words.append(token)
	return len(swear_words)/len(words), len(set(swear_words))/len(set(words))


# will filter out stopwords, non-words and the topic, then show the most frequent and unevenly used words
def biased_words(group1, group2, topic, frequency_weight):
	group1_wordcount = {}
	group2_wordcount = {}
	versus_counts = {}
	total_num_words = 0

	# count every words occurrence from both groups
	for tweet in group1:
		tokens = remove_stopwords(tweet)
		total_num_words += len(tokens)
		for token in remove_non_words(tokens):
			if token != topic:
				if token not in group1_wordcount:
					group1_wordcount[token] = 1
				else:
					group1_wordcount[token] += 1

	for tweet in group2:
		tokens = remove_stopwords(tweet)
		total_num_words += len(tokens)
		for token in remove_non_words(tokens):
			if token != topic:
				if token not in group2_wordcount:
					group2_wordcount[token] = 1
				else:
					group2_wordcount[token] += 1

	max_relative_to_total = {'value': 0}  	# this will be the highest number of occurrences of any word
											# (relative to the total number of words

	# Given the number of occurrences of a word (in each group) this calculates the imbalance of usage
	# (the actual value that is used for the ranking is calculated later, as we need 'max_relative_to_total' for that
	def calc_imbalance(group1_count, group2_count):
		absolute_difference = abs(group1_count - group2_count)
		total_occurrences = group1_count + group2_count
		relative_diff = absolute_difference / total_occurrences
		relative_to_total = total_occurrences / total_num_words
		if max_relative_to_total['value'] < relative_to_total:
			max_relative_to_total['value'] = relative_to_total
		return relative_diff, relative_to_total

	# calculate imbalance for all words in group one
	for word in group1_wordcount:
		occurrences_other_group = 0
		if word in group2_wordcount:
			occurrences_other_group = group2_wordcount[word]
		versus_counts[word] = (group1_wordcount[word], occurrences_other_group, calc_imbalance(occurrences_other_group, group1_wordcount[word]))

	# until now we only have all words in group1, but there are probably some in group2 which didn't occur in group1.
	# so do the same again for all words in group2 which are not already in versus_counts
	for word in group2_wordcount:
		if word not in versus_counts:
			occurrences_other_group = 0
			if word in group1_wordcount:
				occurrences_other_group = group1_wordcount[word]
			versus_counts[word] = (occurrences_other_group, group2_wordcount[word], calc_imbalance(occurrences_other_group, group2_wordcount[word]))

	# the actual imbalance value is calculated here for the ranking
	# we are aware that this is super-inefficient (should only be calculated once), this is just one of the corners
	# we're cutting now to save time
	def compare(item1, item2):
		relative_diff_1 = versus_counts[item1][2][0]
		frequency_1 = versus_counts[item1][2][1]
		score1 = relative_diff_1 + frequency_1 * (frequency_weight / max_relative_to_total['value'])

		relative_diff_2 = versus_counts[item2][2][0]
		frequency_2 = versus_counts[item2][2][1]
		score2 = relative_diff_2 - frequency_2 * (frequency_weight / max_relative_to_total['value'])
		return score1 - score2

	# perform ranking
	versus_keys_sorted = sorted(versus_counts, key=functools.cmp_to_key(compare))

	# return list of tuples (words, imbalance-data)
	vers_count_list = []
	for key in versus_keys_sorted:
		vers_count_list.append((key, versus_counts[key]))
	return vers_count_list


def attitude_towards_top_words(tweets, word_biases, sentiment_function):
	scores = [tweet['score'] for tweet in tweets]
	sent_avg = sum(scores)/float(len(scores))
	import itertools
	n = 0
	p = 0
	n_sent = {}
	p_sent = {}
	limit = 20
	for word in word_biases[::-1]:
		if len(word[0]) != 1:
			if word[1][0] > word[1][1] and n < limit:
				t = [tweet for tweet in tweets if word[0] in tweet['Text']]
				results, t = sentiment_function(word[0], t, False)
				n_sent[word[0]] = results['sentiments']
				n += 1
			elif word[1][0] < word[1][1] and p < limit:
				t = [tweet for tweet in tweets if word[0] in tweet['Text']]
				results, t = sentiment_function(word[0], t, False)
				p_sent[word[0]] = results['sentiments']
				p += 1
	n_values = list(itertools.chain.from_iterable(n_sent.values()))
	n_avg = sum(n_values)/float(len(n_values))
	p_values = list(itertools.chain.from_iterable(p_sent.values()))
	p_avg = sum(p_values)/float(len(p_values))
	return {
		'negative': {
			'avg': n_avg,
			'diff': n_avg-sent_avg,
			'words': {k: sum(v)/float(len(v)) if len(v) != 0 else 0 for k, v in n_sent.items()}
		},
		'positive': {
			'avg': p_avg,
			'diff': p_avg-sent_avg,
			'words': {k: sum(v)/float(len(v)) if len(v) != 0 else 0 for k, v in p_sent.items()}
		}
	}