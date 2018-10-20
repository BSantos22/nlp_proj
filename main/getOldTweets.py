import sys
import json
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
from pprint import pprint

nltk.download('stopwords')

tweets_dir = "results"
raw_dir = tweets_dir + "/filteredOut"
filtered_dir = tweets_dir + "/tweets"
sent_dir = tweets_dir + "/sentiments"
time_format = "%Y-%m-%d"

def tweet_to_json(tweet):
	return {'Text': tweet.text,
			'Retweets': tweet.retweets,
			'Favorites': tweet.favorites,
			'Hashtags': tweet.hashtags,
			'ID': tweet.id,
			'Username': tweet.username,
			'Permalink': tweet.permalink,
			'Mentions': tweet.mentions,
			'Date': str(tweet.date)}


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


def acquireTweets(topic, tweets_per_month, start, fame):
	file_name = topic + "_" + str(tweets_per_month) + "_" + str(fame)

	def tweet_is_fame_dayum(tweet):
		return 2 * int(tweet["Retweets"]) + int(tweet['Favorites']) > fame

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
					d = json.dumps(tweet, sort_keys=True, indent=4)
					print(d, file=open(tweets_file_name_filtered, 'a+'))
			if not used:
				print(d, file=open(tweets_file_name, 'a+'))
		num_tweets += len(tweets)
		print(str(num_tweets) + " tweets, " + str(num_filtered) + " filtered")
		start = until


def sentiment(topic):
	nltk.download('twitter_samples')
	nltk.download('vader_lexicon')

	from nltk.corpus import twitter_samples
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	
	tweets_file_name = filtered_dir + '/' + topic + '.txt'
	print("Looking for data in " + tweets_file_name)
	found_it = os.path.exists(tweets_file_name)
	print("Could" + (" not " if not found_it else " ") + "find it!")
	if found_it:
		tweets = []
		s = open(tweets_file_name).read()
		while s:
			s = s.strip()
			obj, pos = json.JSONDecoder().raw_decode(s)
			if not pos:
				raise ValueError('no JSON object found at %i' % pos)
			tweets.append(obj)
			s = s[pos:]
		sid = SentimentIntensityAnalyzer()
		for tweet in tweets:
			print(tweet['Text'])
			ss = sid.polarity_scores(tweet['Text'])
			tweet['score'] = ss['compound']
		tweets = sorted(tweets, key=lambda k: k['score'])[::-1]
		for tweet in tweets:
			print('{0}: {1}, '.format(tweet['Text'], tweet['score'])) 
	'''
	- load saved tweet data from tweets_fine_name
	- train classifier using twitter namples
	- sentiment analysis
	- also maybe train classifier using sentiment lexicons
	- sentiment analysis, compare! Which performs better (maybe try both?)
	- save positive and negative tweets to different lists
	- look for patterns that only occur in one group
	'''


def getCommand():
	command = input('\"acquire\" or \"sentiment\": ')
	if command == "acquire":
		num_per_month = input('tweets per day: ')
		start_date = input("start date: (yyyy-mm-dd): ")
		min_fame = input("min fame (2 * retweets + favourites): ")
		acquireTweets(tweet_topic, int(num_per_month), parser.parse(start_date), int(min_fame))
	elif command == "sentiment":
		sentiment(tweet_topic)
	else:
		print("what?")
		getCommand()


if __name__ == '__main__':
	if not os.path.exists(tweets_dir):
		os.makedirs(tweets_dir)
	if not os.path.exists(raw_dir):
		os.makedirs(raw_dir)
	if not os.path.exists(filtered_dir):
		os.makedirs(filtered_dir)
	if not os.path.exists(sent_dir):
		os.makedirs(sent_dir)
	while(True):  # yeah, i know
		tweet_topic = input('topic: ')
		#__# getCommand()
		sentiment(tweet_topic)
		print("\n\nCool, done, next:")

