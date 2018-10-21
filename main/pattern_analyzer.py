from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from string import digits
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()
porter_stemmer = PorterStemmer()


def is_link(text):
	link_indicators = ["http", ".com", ".org", ".net", ".co.uk"]
	for indicator in link_indicators:
		if indicator in text:
			return True
	return False


def lexical_diversity(tweets):
	#words_naive = []
	#words_lowered = []
	#words_stemmed = []
	words_lemmas = []
	for tweet in tweets:
		tokens = tokenizer.tokenize(tweet["Text"])
		for token in tokens:
			if not is_link(token):
				remove_digits = str.maketrans('', '', digits)
				remove_signs = str.maketrans('', '', '+*\',.-;:@!"§$%&/()[]{}=?\\_#|~«»–—‘’“”•…')
				clean_token = token.translate(remove_digits).translate(remove_signs)
				if len(clean_token) > 0 and len(clean_token.translate(str.maketrans('', '', ' '))) > 0:
					#words_naive.append(clean_token)
					#words_lowered.append(clean_token.lower())
					#words_stemmed.append(porter_stemmer.stem(clean_token.lower()))
					words_lemmas.append(wordnet_lemmatizer.lemmatize(clean_token.lower()))

	#naive_res = len(set(words_naive))/len(words_naive)
	#lowered_res = len(set(words_lowered))/len(words_lowered)
	#stemmed_res = len(set(words_stemmed))/len(words_stemmed)

	diversity = len(set(words_lemmas)) / len(words_lemmas)

	#print("Naive diversity: "+str(len(set(words_naive)))+" / " + str(len(words_naive)) + " = " + str(naive_res))
	#print("Words lowered: "+str(len(set(words_lowered)))+" / " + str(len(words_lowered)) + " = " + str(lowered_res))
	#print("Stemmed diversity: "+str(len(set(words_stemmed)))+" / " + str(len(words_stemmed)) + " = " + str(stemmed_res))
	print("Lemmatised diversity: "+str(len(set(words_lemmas)))+" / " + str(len(words_lemmas)) + " = " + str(diversity))
	return diversity
