import sys
import json
import shutil
from operator import pos

if sys.version_info[0] < 3:
	import got
else:
	import got3 as got
import os
import nltk
from nltk.corpus import stopwords
import operator
from nltk.tokenize import TweetTokenizer
import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import numpy as np
import pattern_analyzer

nltk.download('stopwords')

tweets_dir = "results"
raw_dir = tweets_dir + "/filteredOut"
filtered_dir = tweets_dir + "/tweets"
sent_dir = tweets_dir + "/sentiments"
anal_dir = tweets_dir + "/analysis"
time_format = "%Y-%m-%d"

edge = 0.2


def tweets_from_file(topic):
	tweets = []
	tweets_file_name = filtered_dir + '/' + topic + '.txt'
	print("Looking for data in " + tweets_file_name)
	if not os.path.exists(tweets_file_name):
		print("Aww man, could not find " + tweets_file_name)
		print("Be sure to acquire data first!")
		return tweets
	s = open(tweets_file_name).read()
	while s:
		s = s.strip()
		obj, pos = json.JSONDecoder().raw_decode(s)
		if not pos:
			raise ValueError('no JSON object found at %i' % pos)
		tweets.append(obj)
		s = s[pos:]
	return tweets


def tweets_from_path(path):
	tweets = []
	print("Looking for data in " + path)
	if not os.path.exists(path):
		print("Aww man, could not find " + path)
		print("Be sure to acquire data first!")
		return tweets
	s = open(path).read()
	while s:
		s = s.strip()
		obj, pos = json.JSONDecoder().raw_decode(s)
		if not pos:
			raise ValueError('no JSON object found at %i' % pos)
		tweets.append(obj)
		s = s[pos:]
	return tweets


def tweet_to_json(tweet):
	return {'Text': tweet['Text'],
			'Retweets': tweet['Retweets'],
			'Favorites': tweet['Favorites'],
			'Hashtags': tweet['Hashtags'],
			'ID': tweet['ID'],
			'Username': tweet['Username'],
			'Permalink': tweet['Permalink'],
			'Mentions': tweet['Mentions'],
			'Date': str(tweet['Date']),
			'Side': ''}


def tweet_is_english(tweet):
	text = tweet['Text']
	languages_ratios = {}
	words = TweetTokenizer().tokenize(text)
	words_lower = TweetTokenizer().tokenize(text.lower())
	words_set = set(words_lower)
	for language in stopwords.fileids():
		stopwords_set = set(stopwords.words(language))
		common_elements = words_set.intersection(stopwords_set)
		languages_ratios[language] = len(common_elements)
	predicted_language = max(languages_ratios.items(), key=operator.itemgetter(1))[0]
	if predicted_language == "english":
		return words
	return None


def acquireTweets(topic, tweets_per_month, start, fame):	# could probs be improved by threading
	file_name = topic + "_" + str(tweets_per_month) + "_" + str(fame)

	def tweet_is_fame_dayum(tweet):
		return 2 * int(tweet["Retweets"]) + int(tweet['Favorites']) >= fame

	end = datetime.datetime.now()
	num_tweets = 0
	num_filtered = 0
	tweets_file_name = raw_dir + '/' + file_name + '.txt'
	tweets_file_name_filtered = filtered_dir + '/' + topic + '.txt'
	while start < end:
		until = start + relativedelta(days=1)
		print(start.strftime(time_format) + " to " + until.strftime(time_format))
		tweet_criteria = got.manager.TweetCriteria().setQuerySearch(topic).setSince(start.strftime(time_format)).setUntil(
			until.strftime(time_format)).setTopTweets(True).setMaxTweets(tweets_per_month)
		tweets = got.manager.TweetManager.getTweets(tweet_criteria)
		for x in range(len(tweets)):
			tweet = tweet_to_json(tweets[x])
			d = json.dumps(tweet, sort_keys=True, indent=4)
			used = False
			if tweet_is_fame_dayum(tweet):
				words_set = tweet_is_english(tweet)
				if words_set is not None:
					used = True
					num_filtered += 1
					tweet["tokens"] = str(words_set)
					print(d, file=open(tweets_file_name_filtered, 'a+'))
			if not used:
				print(d, file=open(tweets_file_name, 'a+'))
		num_tweets += len(tweets)
		print(str(num_tweets) + " tweets, " + str(num_filtered) + " of interest")
		start = until


# At first I put everything between 0.33 and -0.33 as neutral, but some of those are already pretty
# opinionated, for instance: "That Kavanaugh even showed up to that bizarre White House pep rally shows how temperamentally unqualified he is to sit on SCOTUS."
# had a score of only -0.3182.
# For now I put the line at -0.2 and 0.2
def sentiment(topic, tweets):
	results = {
		'TPOS': 0,
		'FPOS(NEU)': 0,
		'FPOS(NEG)': 0,
		'TNEU': 0,
		'FNEU(POS)': 0,
		'FNEU(NEG)': 0,
		'TNEG': 0,
		'FNEG(POS)': 0,
		'FNEG(NEU)': 0,
		"edge": "",
		"std": "",
		"controversial": "",
		"sentiments": []
	}
	nltk.download('vader_lexicon')
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	for tweet in tweets:
		ss = sid.polarity_scores(tweet['Text'])
		tweet['score'] = ss['compound']
		results['sentiments'].append(tweet['score'])
		if tweet['score'] > edge:
			file_name = sent_dir + '/' + topic + '_support_sent.txt'
			if tweet['Side'] == 'pos':
				results['TPOS'] += 1
			elif tweet['Side'] == 'neg':
				results['FPOS(NEG)'] += 1
			else:
				results['FPOS(NEU)'] += 1
		elif tweet['score'] < -edge:
			file_name = sent_dir + '/' + topic + '_against_sent.txt'
			if tweet['Side'] == 'pos':
				results['FNEG(POS)'] += 1
			elif tweet['Side'] == 'neg':
				results['TNEG'] += 1
			else:
				results['FNEG(NEU)'] += 1
		else:
			file_name = sent_dir + '/' + topic + '_neutral_sent.txt'
			if tweet['Side'] == 'pos':
				results['FNEU(POS)'] += 1
			elif tweet['Side'] == 'neg':
				results['FNEU(NEG)'] += 1
			else:
				results['TNEU'] += 1
		d = json.dumps(tweet, sort_keys=True, indent=4)
		print(d, file=open(file_name, 'a+'))
	return results, tweets


def sentiment_results(topic, results, out):
	import matplotlib.pyplot as plt
	binCount = plt.hist(results['sentiments'], bins=[-1, -edge, edge, 1])[0]
	print(binCount)
	plt.hist(results['sentiments'])
	plt.savefig(sent_dir + "/" + topic + "_plot" + "_" + out + ".png")
	plt.clf()
	std = np.std(results['sentiments'])
	controversial = std > edge and abs(binCount[0] - binCount[2]) < len(results['sentiments'])/5
	print("Controversial " + str(controversial))
	results['edge'] = edge
	results['std'] = std
	results['controversial'] = str(controversial)
	print(json.dumps(results), file=open(sent_dir + "/" + topic + "_polarisation" + "_" + out + ".txt", 'w+'))
	return results


def dependency_parser(topic, tweets, out, uses_sentiment):
	import spacy
	nlp = spacy.load('en')
	lexicon = load_vader_lexicon()

	results = {
		'TPOS': 0,
		'FPOS(NEU)': 0,
		'FPOS(NEG)': 0,
		'TNEU': 0,
		'FNEU(POS)': 0,
		'FNEU(NEG)': 0,
		'TNEG': 0,
		'FNEG(POS)': 0,
		'FNEG(NEU)': 0,
		"edge": "",
		"std": "",
		"controversial": "",
		"sentiments": []
	}
	for tweet in tweets:
		doc = nlp(tweet['Text'])
		score = []
		for token in doc:
			try:
				if (topic.upper() in token.text.upper()):
					score.append(lexicon[token.head.text.lower()])
				if (topic.upper() in token.head.text.upper()):
					score.append(lexicon[token.text.lower()])
			except KeyError:
				continue
		results['sentiments'].append(sum([float(i) for i in score]))
		
		if uses_sentiment:
			if not score:
				if tweet['score'] > edge:
					rating = 'neu'
				elif tweet['score'] < -edge:
					rating = 'neu'
				else:
					rating = 'neu'
			elif all(float(i) > 0 for i in score):
				if tweet['score'] > edge:
					rating = 'pos'
				elif tweet['score'] < -edge:
					rating = 'neu'
				else:
					rating = 'neu'
			elif all(float(i) < 0 for i in score):
				if tweet['score'] > edge:
					rating = 'neu'
				elif tweet['score'] < -edge:
					rating = 'neg'
				else:
					rating = 'neg'
			else:
				if tweet['score'] > edge:
					rating = 'pos'
				elif tweet['score'] < -edge:
					rating = 'neg'
				else:
					rating = 'neu'
		else:
			if not score:
				rating = 'neu'
			elif (all(float(i) > 0 for i in score)):
				rating = 'pos'
			elif (all(float(i) < 0 for i in score)):
				rating = 'neg'
			else:
				rating = 'neu'

		if rating == 'neu':
			file_name = sent_dir + '/' + topic + '_neutral_' + out + '.txt'
			if tweet['Side'] == 'pos':
				results['FNEU(POS)'] += 1
			elif tweet['Side'] == 'neg':
				results['FNEU(NEG)'] += 1
			else:
				results['TNEU'] += 1
		elif rating == 'pos':
			file_name = sent_dir + '/' + topic + '_support_' + out + '.txt'
			if tweet['Side'] == 'pos':
				results['TPOS'] += 1
			elif tweet['Side'] == 'neg':
				results['FPOS(NEU)'] += 1
			else:
				results['FPOS(NEU)'] += 1
		elif rating == 'neg':
			file_name = sent_dir + '/' + topic + '_against_' + out + '.txt'
			if tweet['Side'] == 'pos':
				results['FNEG(POS)'] += 1
			elif tweet['Side'] == 'neg':
				results['TNEG'] += 1
			else:
				results['FNEG(NEU)'] += 1
		d = json.dumps(tweet, sort_keys=True, indent=4)
		print(d, file=open(file_name, 'a+'))
	return results


def preprocess_trump_tweets(topic, tweets):
	import re
	import itertools
	for tweet in tweets:
		tokens = tweet['Text'].split(' ')
		# Remove @ and #
		i = 0
		while i < len(tokens):
			if tokens[i] == '@' or tokens[i] == '#':
				del tokens[i]
				i -= 1
			i += 1
		
		# Split words by uppercase
		i = 0
		while i < len(tokens):
			tokens[i] = re.findall('[a-zA-Z][^A-Z]*', tokens[i])
			i += 1
		tokens = list(itertools.chain.from_iterable(tokens))

		# Replace you and u with topic
		for i in range(len(tokens)):
			if topic not in tweet['Text']:
				if tokens[i].lower() == 'you' or tokens[i].lower() == 'u':
					tokens[i] = topic

		# Replace he with topic
		for i in range(len(tokens)):
			if topic in tweet['Text']:
				if tokens[i].lower() == 'he':
					tokens[i] = topic
		
		tweet['Text'] = " ".join(tokens)
	return tweets


def load_vader_lexicon():
	lines = open('./main/vader_lexicon/vader_lexicon.txt').readlines()
	lexicon = {}
	for line in lines:
		lexicon[line.split('\t')[0]] = line.split('\t')[1]
	return lexicon


def analyse(topic, grouping_method):
	topic_dir = anal_dir + "/" + topic
	if os.path.exists(topic_dir):
		shutil.rmtree(topic_dir)
	os.makedirs(topic_dir)
	import nltk
	nltk.download('wordnet')

	analysis_neg = {}
	analysis_pos = {}

	sentiment_files = [
		sent_dir + '/' + topic + '_against_' + grouping_method + '.txt',
		sent_dir + '/' + topic + '_neutral_' + grouping_method + '.txt',
		sent_dir + '/' + topic + '_support_' + grouping_method + '.txt',
		sent_dir + '/' + topic + '_polarisation_' + grouping_method + '.txt'
	]
	print("Looking for sentiment data...")

	for file in sentiment_files:
		if not os.path.exists(file):
			print("Aww man, could not find " + file)
			print("Be sure to do the sentiment analysis first!")
			return
	print("Aight, let's see...")

	negative_tweets = tweets_from_path(sentiment_files[0])
	positive_tweets = tweets_from_path(sentiment_files[2])

	analysis_neg["lexicalDiversity"] = pattern_analyzer.lexical_diversity(negative_tweets)
	analysis_pos["lexicalDiversity"] = pattern_analyzer.lexical_diversity(positive_tweets)

	analysis_neg["profanityShare"] = pattern_analyzer.profanity_share(negative_tweets)
	analysis_pos["profanityShare"] = pattern_analyzer.profanity_share(positive_tweets)

	biases_file = open(topic_dir + "/" + "word_biases.txt", 'w+')

	frequency_weight = input("Weight of frequency (0...1):")

	word_biases = pattern_analyzer.biased_words(negative_tweets, positive_tweets, topic, float(frequency_weight))

	analysis_neg["attitudeTopWords"] = pattern_analyzer.attitude_towards_top_words(negative_tweets, word_biases)
	analysis_pos["attitudeTopWords"] = pattern_analyzer.attitude_towards_top_words(positive_tweets, word_biases)

	index = len(word_biases)
	row = "#rank: " + "word" + (" " * (30 - len("word"))) + "negative" + (" " * (30 - len("negative"))) + "positive" + (" " * (30 - len("positive"))) + "imbalance, relative frequency" + (" " * (30 - len("imbalance, relative frequency")))
	print(row)
	print(row, file=biases_file)
	print("-"*len(row))
	print("-"*len(row), file=biases_file)
	for x in word_biases:
		row = "#"+str(index)+": " + x[0] + (" " * (30 - len(x[0]))) + str(x[1][0]) + (" " * (30 - len(str(x[1][0])))) + str(x[1][1]) + (" " * (30 - len(str(x[1][1])))) + str(x[1][2])
		try:
			print(row)
			print(row, file=biases_file)
		except UnicodeEncodeError:
			print("Could not save")
			print(row)
		index -= 1

	d = json.dumps(analysis_neg, sort_keys=True, indent=4)
	print(d, file=open(topic_dir + "/" + "neg.txt", 'w+'))
	d = json.dumps(analysis_pos, sort_keys=True, indent=4)
	print(d, file=open(topic_dir + "/" + "pos.txt", 'w+'))


def getCommand(topic):
	command = input('1: acquire, 2: sentiment, 3: sentiment w/ preprocessing, 4: dependency, 5: dependency w/ preprocessing, 6: sentiment & dependency, 7: analyse\n')
	if command == "1":
		num_per_month = input('tweets per day: ')
		start_date = input("start date: (yyyy-mm-dd): ")
		min_fame = input("min fame (2 * retweets + favourites): ")
		acquireTweets(topic, int(num_per_month), parser.parse(start_date), int(min_fame))
	elif command == "2":
		tweets = tweets_from_file(topic)
		results, tweets = sentiment(topic, tweets)
		sentiment_results(topic, results, "sent")
	elif command == "3":
		tweets = tweets_from_file(topic)
		tweets = preprocess_trump_tweets(topic, tweets)
		results, tweets = sentiment(topic, tweets)
		sentiment_results(topic, results, "sent_pp")	
	elif command == "4":
		tweets = tweets_from_file(topic)
		results = dependency_parser(topic, tweets, "dep", False)
		sentiment_results(topic, results, "dep")
	elif command == "5":
		tweets = tweets_from_file(topic)
		tweets = preprocess_trump_tweets(topic, tweets)
		results = dependency_parser(topic, tweets, "dep_pp", False)
		sentiment_results(topic, results, "dep_pp")
	elif command == "6":
		tweets = tweets_from_file(topic)
		results, tweets = sentiment(topic, tweets)
		tweets = preprocess_trump_tweets(topic, tweets)
		results = dependency_parser(topic, tweets, "sent_dep", True)
		sentiment_results(topic, results, "sent_dep")
		return
	elif command == "7":
		grouping_method = input('grouping method: ')
		analyse(topic, grouping_method)
	else:
		print("what?")
		getCommand(topic)


def start():
	if not os.path.exists(tweets_dir):
		os.makedirs(tweets_dir)
	if not os.path.exists(raw_dir):
		os.makedirs(raw_dir)
	if not os.path.exists(filtered_dir):
		os.makedirs(filtered_dir)
	if not os.path.exists(sent_dir):
		os.makedirs(sent_dir)
	if not os.path.exists(anal_dir):
		os.makedirs(anal_dir)
	while(True):  # yeah, i know
		topic = input('topic: ')
		getCommand(topic)
		print("\n\nCool, done, next:")
