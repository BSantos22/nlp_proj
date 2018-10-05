import sys
import json
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got
import os

outputDir = "search_results"

def main(topic):
	def printTweet(t):
		return {'Retweets': t.retweets,
				'Text': t.text,
				'Mentions': t.mentions,
				'Hashtags': t.hashtags,
				'Date': str(t.date)}

	# Example 2 - Get tweets by query search
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(topic).setSince('2015-05-01').setUntil('2015-09-30').setMaxTweets(100)
	fileName = outputDir + '/got_' + topic + '.txt'
	open(fileName, 'w+')
	for x in range(100):
		tweet = printTweet(got.manager.TweetManager.getTweets(tweetCriteria)[x])
		print(json.dumps(tweet, sort_keys=True, indent=4), file=open(fileName, 'a'))


if __name__ == '__main__':
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
	main(input('topic: '))

