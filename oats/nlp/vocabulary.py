import pandas as pd
import numpy as np
import itertools
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

from oats.utils.utils import flatten








def get_vocabulary_from_tokens(tokens):
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








def get_overrepresented_tokens(interesting_text, background_text, max_features):
	"""
	https://liferay.de.dariah.eu/tatom/feature_selection.html
	This way uses the difference in the rate of each particular words between the 
	interesting text and the background text to determine what the vocabulary of 
	relevant words should be. This means we are selecting as features things that are 
	important in some paritcular domain but are not as important in the general
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








def reduce_vocabulary_connected_components(descriptions, tokens, distance_matrix, threshold):
	"""
	Reduces the vocabulary size for a dataset of provided tokens by looking at a provided 
	distance matrix between all the words and creating new tokens to represent groups of 
	words that have a small distance (less than the threshold) between two of the members
	of that group. This problem is solved here as a connected components problem by creating
	a graph where tokens are words, and each word is connected to itself and any word where
	the distance to that word is less than the threshold. Note that the Linares Pontes is 
	generally favorable to this approach because if the threshold is too high the connected
	components can quickly become very large.
	
	Args:
		descriptions (TYPE): Description
		tokens (list): A list of tokens from which to construct the vocabulary.
		distance_matrix (np.array): An by n square matrix of distances where n must be length of tokens list and indices must correspond.
		threshold (float): The value where a distance of less than this threshold indicates.
		the words should be collapsed to a new token.
	
	Returns:
		dict: Mapping between IDs and text descriptions with reduced vocabulary, matches input.
		dict: Mapping between tokens present in the reduced vocab and lists of corresponding original vocabulary tokens.
	"""

	g = nx.Graph()
	edges = []
	node_id_to_token = {str(i):token for i,token in enumerate(tokens)}
	token_to_node_id = {token:str(i) for i,token in node_id_to_token.items()}

	# Add the self edges between each token, then edges that satisfy the distance threshold.
	edges.extend([(token_to_node_id[token],token_to_node_id[token]) for token in tokens]) 
	for i,j in itertools.product(np.arange(distance_matrix.shape[0]),np.arange(distance_matrix.shape[0])):
		if distance_matrix[i,j] < threshold:
			edges.append((token_to_node_id[tokens[i]],token_to_node_id[tokens[j]]))   

	# Add all the edges from the graph and build the dictionaries by looking at connected components.
	g.add_edges_from(edges)
	components = nx.connected_components(g)
	transform_dict = {}
	untransform_dict = defaultdict(list)
	new_token_ctr = 0
	for c in components:
		if len(c) > 1:
			for node_id in c:
				word = node_id_to_token[node_id]
				new_token = "TOKEN_{}".format(new_token_ctr)
				transform_dict[word] = new_token
				untransform_dict[new_token].append(word)
			new_token_ctr = new_token_ctr+1
		else:
			word = node_id_to_token[list(c)[0]]
			transform_dict[word] = word
			untransform_dict[word] = [word]
	
	# Do the replacements in each input description and return the modified dictionary of them.
	reduced_descriptions = {}
	for i,description in descriptions.items():
		reduced_description = " ".join([transform_dict[token] for token in description.split()])
		reduced_descriptions[i] = reduced_description
	return(reduced_descriptions, transform_dict ,untransform_dict)













def reduce_vocabulary_linares_pontes(descriptions, tokens, distance_matrix, n):
	"""
	Implementation of the algorithm described in this paper. In short, this returns the descriptions
	with each word replaced by the most frequently used token in the set of tokens that consists of 
	that word and the n most similar words as given by the distance matrix provided. Some values of 
	n that are used in the papers are 1, 2, and 3. Note that the descriptions in the passed in 
	dictionary should already be preprocessed in whatever way is necessary, but they should atleast
	be formatted as lowercase tokens that are separated by a single space in each description. The
	tokens in the list of tokens should be pulled directly from those descriptions and be found
	by splitting by a single space. They are passed in as a separate list thought because the index
	of the token in the list has to correspond to the index of that token in the distance matrix. 
	The descriptions should not contain any tokens which are not present in the tokens list.

	Elvys Linhares Pontes, Stéphane Huet, Juan-Manuel Torres-Moreno, Andréa Carneiro Linhares. 
	Automatic Text Summarization with a Reduced Vocabulary Using Continuous Space Vectors. 
	21st International Conference on Applications of Natural Language to Information Systems (NLDB),
	2016, Salford, United Kingdom. pp.440-446, ff10.1007/978-3-319-41754-7_46ff. ffhal-01779440
	
	Args:
		descriptions (dict): A mapping between IDs and text descriptions.
		tokens (list): A list of strings which are tokens that appear in the descriptions. 
		distance_matrix (np.array): A square array of distances between the ith and jth token in the tokens list. 
		n (int): The number of most similar words to consider when replacing a word in building the reduced vocabulary.
	
	Returns:
		dict: Mapping between IDs and text descriptions with reduced vocabulary, matches input.
		dict: Mapping between tokens present in the original vocab and the token it is replaced with in the reduced vocabulary.
		dict: Mapping between tokens present in the reduced vocab and lists of corresponding original vocabulary tokens.
	"""

	# Find the frequency distribution of all of the tokens in the passed in descriptions.
	fdist = FreqDist(flatten([[token for token in description.split()] for description in descriptions.values()]))
	token_to_index = {token:i for i,token in enumerate(tokens)}
	index_to_token = {i:token for i,token in enumerate(tokens)}
	
	# Create a mapping between all the tokens and the words they'll be replaced with.
	token_to_reduced_vocab_token = {}
	reduced_vocab_token_to_tokens = defaultdict(list)
	for token in tokens:
		index = token_to_index[token]
		n_indices = np.argpartition(distance_matrix[index], n)[:n]
		n_tokens = [index_to_token[idx] for idx in list(n_indices)]
		n_frequencies = [fdist[token] for token in n_tokens]
		maxfreq_token = n_tokens[np.argmax(n_frequencies)]
		token_to_reduced_vocab_token[token] = maxfreq_token
		reduced_vocab_token_to_tokens[maxfreq_token].append(token)

		
	# Do the replacements in each input description and return the modified dictionary of them.
	reduced_descriptions = {}
	for i,description in descriptions.items():
		reduced_description = " ".join([token_to_reduced_vocab_token[token] for token in description.split()])
		reduced_descriptions[i] = reduced_description
	return(reduced_descriptions, token_to_reduced_vocab_token, reduced_vocab_token_to_tokens)
			


	











