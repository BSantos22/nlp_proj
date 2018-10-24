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


def is_link(text):
	link_indicators = ["http", ".com", ".org", ".net", ".co.uk"]
	for indicator in link_indicators:
		if indicator in text:
			return True
	return False


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


def remove_stopwords(tweet):
	lowered_words = [x.lower() for x in tokenizer.tokenize(tweet["Text"])]
	return [x for x in lowered_words if x not in stops]


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


# will filter out stopwords, non-words and the topic (as it throws off the calculation, it's kinda all over the place)
def biased_words(group1, group2, topic, frequency_weight):
	group1_wordcount = {}
	group2_wordcount = {}
	versus_counts = {}
	total_num_words = 0
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

	# now we have the wordcounts for each group

	max_relative_to_total = {'value': 0}

	def calc_bias(group1_count, group2_count):
		absolute_difference = abs(group1_count - group2_count)
		total_occurrences = group1_count + group2_count
		relative_diff = absolute_difference / total_occurrences
		relative_to_total = total_occurrences / total_num_words
		if max_relative_to_total['value'] < relative_to_total:
			max_relative_to_total['value'] = relative_to_total
		return relative_diff, relative_to_total

	for word in group1_wordcount:
		occurrences_other_group = 0
		if word in group2_wordcount:
			occurrences_other_group = group2_wordcount[word]
		versus_counts[word] = (group1_wordcount[word], occurrences_other_group, calc_bias(occurrences_other_group, group1_wordcount[word]))

	# until now we only have all words in group1, but there are probably some in group2 which didn't occur in group1.
	# so do the same again for all words in group2 which are not already in versus_counts
	for word in group2_wordcount:
		if word not in versus_counts:
			occurrences_other_group = 0
			if word in group1_wordcount:
				occurrences_other_group = group1_wordcount[word]
			versus_counts[word] = (occurrences_other_group, group2_wordcount[word], calc_bias(occurrences_other_group, group2_wordcount[word]))

	def compare(item1, item2):
		return versus_counts[item1][2][0]+versus_counts[item1][2][1] * (frequency_weight / max_relative_to_total['value']) - versus_counts[item2][2][0] - versus_counts[item2][2][1] * (frequency_weight / max_relative_to_total['value'])

	# problem with this: a difference of 100 is not significant if the word is mentioned thousands of times on both sides...
	versus_keys_sorted = sorted(versus_counts, key=functools.cmp_to_key(compare))

	vers_count_list = []
	for key in versus_keys_sorted:
		vers_count_list.append((key, versus_counts[key]))
	return vers_count_list


def attitude_towards_top_words(tweets, word_biases):
	return ""