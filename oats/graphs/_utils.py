from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.neighbors import DistanceMetric
from itertools import product
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import functools
import pronto
import os
import sys
import glob
import math
import re
import random
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



from oats.nlp.search import binary_search_rabin_karp
from oats.utils.utils import flatten





def _square_adjacency_matrix_to_edgelist(matrix, indices_to_ids):
	"""
	Convert the matrix to a dataframe that specifies the nodes and edges of a graph.
	Additionally a dictionary mapping indices in the array to node names (integers) 
	is passed in because the integers that refer to the position in the array do not
	necessarily have to be the integers that are used as the node IDs in the graph
	that is specified by the resulting list of edges. This is intended for handling
	square arrays where the rows and columns are referring to the identical sets of 
	nodes.

	Args:
		matrix (numpy array): A square array which is considered an adjacency matrix.
		indices_to_ids (dict): Mapping between indices of the array and node names.
	
	Returns:
		pandas.Dataframe: Dataframe where each row specifies an edge in a graph.
	"""

	df_of_matrix = pd.DataFrame(matrix)									# Convert the numpy array to a pandas dataframe.
	boolean_triu = np.triu(np.ones(df_of_matrix.shape)).astype(np.bool)	# Create a boolean array of same shape where upper triangle is true.
	df_of_matrix = df_of_matrix.where(boolean_triu)						# Make everything but the upper triangle NA so it is ignored by stack.
	melted_matrix = df_of_matrix.stack().reset_index()					# Melt (stack) the array so the first two columns are matrix indices.
	melted_matrix.columns = ["from", "to", "value"]						# Rename the columns to indicate this specifies a graph.
	melted_matrix["from"] = pd.to_numeric(melted_matrix["from"])		# Make sure node names are integers because IDs have to be integers.
	melted_matrix["to"] = pd.to_numeric(melted_matrix["to"])			# Make sure node names are integers because IDs have to be integers.
	melted_matrix["from"] = melted_matrix["from"].map(indices_to_ids)	# Rename the node names to be IDs from the dataset not matrix indices.
	melted_matrix["to"] = melted_matrix["to"].map(indices_to_ids)		# Rename the node names to be IDS from the dataset not matrix indices.
	return(melted_matrix)												# Return the melted matrix that looks like an edge list.







def _rectangular_adjacency_matrix_to_edgelist(matrix, row_indices_to_ids, col_indices_to_ids):
	"""
	Convert the matrix to a dataframe that specifies the nodes and edges of a graph.
	Additionally two dictionaries mapping indices in the array to node names (integers)
	are passed in because the integers that refer to the position in the array do not 
	necessarily have to be the integers that are used as the node IDs in the graph that
	is specified by the resulting list of edges. This is intended for rectangular arrays
	where the nodes represented by each column are different from the nodes represented 
	by each row. 

	Args:
		matrix (numpy array): A rectangular array which is considered an adjacency matrix.
		row_indices_to_ids (dict): Mapping between indices of the array and node names.
		col_indices_to_ids (dict): Mapping between indices of the array and node names.
	
	Returns:
		pandas.Dataframe: Dataframe where each row specifies an edge in a graph.
	"""
	df_of_matrix = pd.DataFrame(matrix)										# Convert the numpy array to a pandas dataframe.
	melted_matrix = df_of_matrix.stack().reset_index()						# Melt (stack) the array so the first two columns are matrix indices.
	melted_matrix.columns = ["from", "to", "value"]							# Rename the columns to indicate this specifies a graph.
	melted_matrix["from"] = pd.to_numeric(melted_matrix["from"])			# Make sure node names are integers because IDs have to be integers.
	melted_matrix["to"] = pd.to_numeric(melted_matrix["to"])				# Make sure node names are integers because IDs have to be integers.
	melted_matrix["from"] = melted_matrix["from"].map(row_indices_to_ids)	# Rename the node names to be IDs from the dataset not matrix indices.
	melted_matrix["to"] = melted_matrix["to"].map(col_indices_to_ids)		# Rename the node names to be IDS from the dataset not matrix indices.
	return(melted_matrix)													# Return the melted matrix that looks like an edge list.









def _strings_to_count_vectors(*strs, **kwargs):
	"""
	https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
	What are the important keyword arguments that can be passed to the count vectorizer to specify how the 
	features to be counted are selected, and specify how each of them is counted as well? Included here for 
	quick reference for what araguments can be passed in.

	lowercase : boolean, True by default
	analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
	stop_words : string {‘english’}, list, or None (default)
	token_pattern : string
	ngram_range : tuple (min_n, max_n)
	analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
	max_df : float in range [0.0, 1.0] or int, default=1.0
	min_df : float in range [0.0, 1.0] or int, default=1
	max_features : int or None, default=None
	vocabulary : Mapping or iterable, optional
	binary : boolean, default=False

	# Attributes
	vocabulary_: mapping between terms and feature indices
	stop_words_: set of terms that were considered stop words
	"""
	text = [t for t in strs]
	vectorizer = CountVectorizer(text, **kwargs)
	vectorizer.fit(text)
	vectors = vectorizer.transform(text).toarray()
	return(vectors, vectorizer)





def _strings_to_tfidf_vectors(*strs, **kwargs):
	"""
	https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
	What are the important keyword arguments that can be passed to the TFIDF (term-frequency inverse-
	document-frequency) vectorizer to specify how the features to be quantified are selected, and how each
	of them is quantified as well? Incluced here for quick reference for what arguments can be passed in.

	# Arguments
	lowercase : boolean, True by default
	analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
	stop_words : string {‘english’}, list, or None (default)
	token_pattern : string
	ngram_range : tuple (min_n, max_n)
	analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
	max_df : float in range [0.0, 1.0] or int, default=1.0
	min_df : float in range [0.0, 1.0] or int, default=1
	max_features : int or None, default=None
	vocabulary : Mapping or iterable, optional
	binary : boolean, default=False

	# Attributes
	vocabulary_: mapping between terms and feature indices
	idf_: the inverse document frequency vector used in weighting
	stop_words_: set of terms that were considered stop words
	"""
	text = [t for t in strs]
	vectorizer = TfidfVectorizer(text, **kwargs)
	vectorizer.fit(text)
	vectors = vectorizer.transform(text).toarray()
	return(vectors, vectorizer)










# These five methods are all used for storing in SquarePairwiseDistances objects to be able to generate appropriate vectors given new text descriptions.

# These are also used within the context of the other methods to generate vectors given nlp models and parameter choices.
def _infer_document_vector_from_bert(text, model, tokenizer, method="sum", layers=4):
	"""
	This function uses a pretrained BERT model to infer a document level vector for a collection 
	of one or more sentences. The sentence are defined using the nltk sentence parser. This is 
	done because the BERT encoder expects either a single sentence or a pair of sentences. The
	internal representations are drawn from the last n layers as specified by the layers argument, 
	and represent a particular token but account for the context that it is in because the entire
	sentence is input simultanously. The vectors for the layers can concatentated or summed 
	together based on the method argument. The vector obtained for each token then are averaged
	together to for the document level vector.
	
	Args:
	    text (str):  A string representing the text for a single node of interest.
	    model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
	    tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
	    method (str): A string indicating how layers for a token should be combined (concat or sum).
	    layers (int): An integer saying how many layers should be used for each token.
	
	Returns:
	    numpy.Array: A numpy array which is the vector embedding for the passed in text. 
	
	Raises:
	    ValueError: The method argument has to be either 'concat' or 'sum'.
	"""

	sentences = sent_tokenize(text)
	token_vecs_cat = []
	token_vecs_sum = []

	for text in sentences:
		marked_text = "{} {} {}".format("[CLS]",text,"[SEP]")
		tokenized_text = tokenizer.tokenize(marked_text)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensor = torch.tensor([segments_ids])
		with torch.no_grad():
			encoded_layers,_ = model(tokens_tensor,segments_tensor)
		token_embeddings = torch.stack(encoded_layers, dim=0)
		token_embeddings = token_embeddings.permute(1,2,0,3)
		batch = 0
		for token in token_embeddings[batch]:
			concatenated_layer_vectors = torch.cat(tuple(token[-layers:]), dim=0)
			summed_layer_vectors = torch.sum(token[-layers:], dim=0)
			token_vecs_cat.append(np.array(concatenated_layer_vectors))
			token_vecs_sum.append(np.array(summed_layer_vectors))

	# Check to make sure atleast one token was found with an embedding to use as a the 
	# vector representation. If there wasn't found, this is because of the combination
	# of what the passed in description was, and how it was handled by either the sentence
	# tokenizing step or the BERT tokenizer methods. Handle this by generating a random
	# vector. This makes the embedding meaningless but prevents multiple instances that
	# do not have embeddings from clustering together in downstream analysis. An expected
	# layer size is hardcoded for this section based on the BERT architecture.
	expected_layer_size = 768
	if len(token_vecs_cat) == 0:
		print("no embeddings found for input text '{}', generating random vector".format(description))
		random_concat_vector = np.random.rand(expected_layer_size*layers)
		random_summed_vector = np.random.rand(expected_layer_size)
		token_vecs_cat.append(random_concat_vector)
		token_vecs_sum.append(random_summed_vector)

	# Average the vectors obtained for each token across all the sentences present in the input text.
	if method == "concat":
		embedding = np.mean(np.array(token_vecs_cat),axis=0)
	elif method == "sum":
		embedding = np.mean(np.array(token_vecs_sum),axis=0)
	else:
		raise ValueError("method argument is invalid")
	return(embedding)









def _infer_document_vector_from_word2vec(text, model, method):
	"""docstring
	"""
	words = text.lower().split()
	words_in_model_vocab = [word for word in words if word in model.wv.vocab]
	if len(words_in_model_vocab) == 0:
		words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
	stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
	if method == "mean":
		vector = stacked_vectors.mean(axis=0)
	elif method == "max":
		vector = stacked_vectors.max(axis=0)
	else:
		raise Error("method argument is invalid")
	return(vector)










def _infer_document_vector_from_doc2vec(text, model):
	"""docstring
	"""
	vector = model.infer_vector(text.lower().split())
	return(vector)





# These two are not actually called from the other methods, they are only used for remembering vectorization scheme for future queries.
def _get_ngrams_vector(text, countvectorizer):
	"""docstring
	"""
	vector = countvectorizer.transform([text]).toarray()[0]
	return(vector)







def _get_annotations_vector(term_list, countvectorizer, ontology):
	"""docstring
	"""
	term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
	term_list = flatten(term_list)
	term_list = list(set(term_list))
	joined_term_string = " ".join(term_list).strip()
	vector = countvectorizer.transform([joined_term_string]).toarray()[0]
	return(vector)


























