import pandas as pd
import numpy as np
import itertools
import networkx as nx
from nltk.tokenize import sent_tokenize
from collections import defaultdict
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







def collapse_vocabulary_by_distance(tokens, distance_matrix, threshold):
	"""
	Reduces the vocabulary size for a dataset of provided tokens by looking at a provided 
	distance matrix between all the words and creating new tokens to represent groups of 
	words that have a small distance (less than the threshold) between two of the members
	of that group. This problem is solved here as a connected components problem by creating
	a graph where tokens are words, and each word is connected to itself and any word where
	the distance to that word is less than the threshold.

	Args:
		tokens (list): A list of tokens from which to construct the vocabulary.
		distance_matrix (n by n square matrix of distances where n must be length of tokens 
		list and indices must correspond): 
		threshold (float): The value where a distance of less than this threshold indicates 
		the words should be collapsed to a new token.
		 
	Returns:
		dict: Vocabulary dictionary mapping tokens to integers zero to n.
		dict: Transforming dictionary mapping tokens of the input list to new collapsed vocabulary tokens, i.e. {"dog"-->"TOKEN1", "cat"-->"TOKEN1"}
		dict: Untransforming dictionary mapping collapsed tokens to lists of tokens in uncollapsed vocabulary, i.e. {"TOKEN1"-->["cat","dog",...]}
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
				new_token = "TOKEN{}".format(new_token_ctr)
				transform_dict[word] = new_token
				untransform_dict[new_token].append(word)
			new_token_ctr = new_token_ctr+1
		else:
			word = node_id_to_token[list(c)[0]]
			transform_dict[word] = word
			untransform_dict[word] = [word]
	vocabulary = {token:i for i,token in enumerate(list(untransform_dict.keys()))}
	return(vocabulary, transform_dict, untransform_dict)













	