import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer










def get_overrepresented_tokens(interesting_text, background_text, max_features):
	"""
	https://liferay.de.dariah.eu/tatom/feature_selection.html
	This way uses the difference in the rate of each particular words between the 
	interesting text and the background text to determine what the vocabulary of 
	relevant words should be. This means we are selecting as features things that
	are important in some paritcular domain but are not as important in the general
	language. This is potentially one method of finding words which will be of removing
	the general words from the text that is parsed for a particular domain. Potential 
	problem is that we actually want words (features) that are good at differentiating 
	different phenotypes, which is a slightly different question. 
	
	Args:
	    interesting_text (str): A string of many tokesn coming form examples of interest.
	    background_text (str): A string of many tokens coming from some background examples.
	    max_features (int): The maximum number of features (tokens) in the returned vocabulary.
	
	Returns:
	    list: A list of features which are tokens, or words, which are represent
	"""
	vectorizer = CountVectorizer(input='content')
	dtm = vectorizer.fit_transform([interesting_text,background_text])
	vocab = np.array(vectorizer.get_feature_names())
	dtm = dtm.toarray()
	rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)
	keyness = rates[0]-rates[1]
	ranking = np.argsort(keyness)[::-1]
	vocab_tokens = vocab[ranking][0:max_features]
	return(vocab_tokens)








def build_vocabulary_from_tokens(tokens):
	"""
	Generates a mapping between each token and some indices 0 to n that can place
	that token at a particular index within a vector. This is a vocabulary dict 
	that is in the format necessary for passing as an argument to the sklearn
	classes for generating feature vectors from input text.
	
	Args:
	    tokens (list): A list of tokens that belong in the vocabulary.
	
	Returns:
	    dict: A mapping between each token and an index from zero to n.
	"""
	vocab_dictionary = {token:i for i,token in enumerate(list(tokens))}
	return(vocab_dictionary)


