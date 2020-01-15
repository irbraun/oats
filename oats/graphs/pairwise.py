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
from oats.graphs.pwgraph import PairwiseGraph






""" Description of the distance functions provided here. 

Category 1: pairwise_square_[method](...)
These functions take a single dictionary mapping IDs to text or similar objects and finds 
the pairwise distances between all of elements in that dictionary using whatever the method
is that is specific to that function. This produces a square matrix of distances that is 
symmetrical along the diagonal.

Category 2: pairwise_rectangular_[method](...)
These functions take two different dicitonaries mappings IDs to text or similar objects and 
finds the pairwise distances between all combinations of an element from one group and an
element from the other group, making the calculation in a way specific whatever the method 
or approach for that function is. This produces a rectangular matrix of distances. The rows
of that matrix correspond to the elements form the first dictionary, and the columns to the
elements from the second dictionary. In edgelist form, the "from" column refers to IDs from
the first dictionary, and the "to" column refers to IDs from the second dictionary. 

Categery 3: elemwise_list_[method](...)
These functions take two lists of text or similar objects that are of the exact same length
and returns a list of distance values calculated based on the method or approach for that 
particular function. The distance value in position i of the returned list is the distance 
found between the element at position i in the first list and the element at position i in 
the second list.
"""






def strings_to_count_vectors(*strs, **kwargs):
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


def strings_to_tfidf_vectors(*strs, **kwargs):
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


def strings_to_vectors(*strs, tfidf=False, **kwargs):
	if tfidf:
		return(strings_to_tfidf_vectors(*strs, **kwargs))
	else:
		return(strings_to_count_vectors(*strs, **kwargs))





















def square_adjacency_matrix_to_edgelist(matrix, indices_to_ids):
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



def rectangular_adjacency_matrix_to_edgelist(matrix, row_indices_to_ids, col_indices_to_ids):
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








































def pairwise_square_doc2vec(model, ids_to_texts, metric):
	"""
	Find distance between strings of text in some input data using Doc2Vec. Note that only 
	very simple preprocessing is done here (case normalizing and splitting on whitespace)
	so any preprocessing steps on the text strings should be done prior to passing them in
	a dictionary to this function.
	
	Args:
	    model (gensim.models.doc2vec): An already loaded Doc2Vec model from a file or training.
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine). 
	
	Returns:
	   	oats.pairwise.PairwiseGraph: Distance matrix and accompanying information.
	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		inferred_vector = model.infer_vector(description.lower().split())
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix
	
	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	
	# Create and return a PairwiseGraph object containing the edgelist, matrix, and dictionaries.
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, 
		id_to_index_in_matrix, 
		id_to_index_in_matrix, 
		index_in_matrix_to_id,
		index_in_matrix_to_id,
		matrix))




def pairwise_square_word2vec(model, ids_to_texts, metric, method="mean"):
	"""	
	Find distance between strings of text in some input data using Word2Vec. Note that only 
	very simple preprocessing is done here (case normalizing and splitting on whitespace)
	so any preprocessing steps on the text strings should be done prior to passing them in
	a dictionary to this function. Note that if no words in a description are in the model 
	vocabulary, then a random word will be selected to represent the text. This avoids using 
	one default value which will force all these descriptions to cluster, and prevents an 
	error being raised due to no vector appearing. This should rarely or never happen as 
	long as the text has been preprocessed into reasonable tokens and the model is large.
	
	Args:
	    model (gensim.models.word2vec): An already loaded Word2Vec model from a file or training.
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine). 
	    method (str, optional): Should the word embeddings be combined with mean or max.
	
	Returns:
	   	oats.pairwise.PairwiseGraph: Distance matrix and accompanying information.
	
	Raises:
	    Error: The 'method' argument has to be one of "mean" or "max".
	"""
	
	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		words = description.lower().split()
		words_in_model_vocab = [word for word in words if word in model.wv.vocab]
		if len(words_in_model_vocab) == 0:
			words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
		stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
		index_in_matrix = len(vectors)
		if method == "mean":
			vectors.append(stacked_vectors.mean(axis=0))
		elif method == "max":
			vectors.append(stacked_vectors.max(axis=0))
		else:
			raise Error("method argument is invalid")
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	
	# Create and return a PairwiseGraph object containing the edgelist, matrix, and dictionaires.
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, 
		id_to_index_in_matrix, 
		id_to_index_in_matrix, 
		index_in_matrix_to_id,
		index_in_matrix_to_id,
		matrix))





def pairwise_square_bert(model, tokenizer, ids_to_texts, metric, method, layers):
	"""
	Find distance between strings of text in some input data using Doc2Vec. The preprocessing
	done to the text strings here is complex, and uses the passed in tokenizer object as well.
	For this reason, in most cases the text passed in to this method should be the raw
	relatively unprocessed sentence of interest. Splitting up of multiple sentences is handled
	in the helper function for this function.
	
	Args:
	    model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
	    tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    method (str): A string indicating how layers for a token should be combined (concat or sum).
	    layers (int): An integer saying how many layers should be used for each token.
	
	Returns:
	   	oats.pairwise.PairwiseGraph: Distance matrix and accompanying information.

	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		inferred_vector = _infer_document_vector_from_bert(model, tokenizer, description, method, layers)
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}

	# Create and return a PairwiseGraph object containing the edgelist, matrix, and dictionaires.
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, 
		id_to_index_in_matrix, 
		id_to_index_in_matrix, 
		index_in_matrix_to_id,
		index_in_matrix_to_id,
		matrix))

def _infer_document_vector_from_bert(model, tokenizer, description, method="sum", layers=4):
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
	    model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
	    tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
	    description (str): A string representing the text for a single node of interest.
	    method (str): A string indicating how layers for a token should be combined (concat or sum).
	    layers (int): An integer saying how many layers should be used for each token.
	
	Returns:
	    numpy.Array: A numpy array which is the vector embedding for the passed in text. 
	
	Raises:
	    ValueError: The method argument has to be either 'concat' or 'sum'.
	"""

	sentences = sent_tokenize(description)
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





def pairwise_square_ngrams(ids_to_texts, metric, tfidf=False, **kwargs):
	"""
	Find distance between strings of text in some input data using n-grams. Note that only 
	very simple preprocessing is done after this point (splitting on whitespace only) so 
	all processing of the text necessary should be done prio to passing to this function.
	
	Args:
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    tfidf (bool, optional): Whether to use TFIDF weighting or not.
	    **kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer()
	
	Returns:
	    oats.pairwise.PairwiseGraph: Distance matrix and accompanying information.
	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	descriptions = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix


	# Apply distance metric over all the vectors to yield a matrix.
	vectors,vectorizer = strings_to_vectors(*descriptions, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	
	# Create and return a PairwiseGraph object containing the edgelist, matrix, and dictionaires.
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, vectorizer, 
		id_to_index_in_matrix, 
		id_to_index_in_matrix, 
		index_in_matrix_to_id,
		index_in_matrix_to_id,
		matrix))




def pairwise_square_annotations(ids_to_annotations, ontology, metric, tfidf=False, **kwargs):
	"""
	Find distance between nodes of interest in the input dictionary based on the overlap in the
	ontology terms that are mapped to those nodes. The input terms for each ID are in the format
	of lists of term IDs. All inherited terms by all these terms will be added in this function 
	using the provided ontology object so that each node will be represented by the union of all 
	the terms inherited by the terms annotated to it. After that step, the term IDs are simply 
	treated as words in a vocabulary, and the same approach as with n-grams is used to generate 
	the distance matrix.
	
	Args:
	    ids_to_annotations (dict): A mapping between IDs and a list of ontology term ID strings.
	    ontology (Ontology): Ontology object with all necessary fields.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    tfidf (bool, optional): Whether to use TFIDF weighting or not.
	    **kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer()
	
	Returns:
	    oats.pairwise.PairwiseGraph: Distance matrix and accompanying information.

	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	joined_term_strings = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,term_list in ids_to_annotations.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		index_in_matrix = len(joined_term_strings)
		joined_term_strings.append(joined_term_string)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	vectors,vectorizer = strings_to_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}

	# Create and return a PairwiseGraph object containing the edgelist, matrix, and dictionaires.
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, vectorizer, 
		id_to_index_in_matrix, 
		id_to_index_in_matrix, 
		index_in_matrix_to_id,
		index_in_matrix_to_id,
		matrix))






























def pairwise_rectangular_doc2vec(model, ids_to_texts_1, ids_to_texts_2, metric):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		inferred_vector = model.infer_vector(description.lower().split())
		vectors.append(inferred_vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		inferred_vector = model.infer_vector(description.lower().split())
		vectors.append(inferred_vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None, 
		id_to_row_index_in_matrix, 
		id_to_col_index_in_matrix, 
		row_index_in_matrix_to_id, 
		col_index_in_matrix_to_id, 
		matrix))



def pairwise_rectangular_word2vec(model, ids_to_texts_1, ids_to_texts_2, metric, method="mean"):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		words = description.lower().split()
		words_in_model_vocab = [word for word in words if word in model.wv.vocab]
		if len(words_in_model_vocab) == 0:
			words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
		stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
		if method == "mean":
			vectors.append(stacked_vectors.mean(axis=0))
		elif method == "max":
			vectors.append(stacked_vectors.max(axis=0))
		else:
			raise Error("method argument is invalid")
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		words = description.lower().split()
		words_in_model_vocab = [word for word in words if word in model.wv.vocab]
		if len(words_in_model_vocab) == 0:
			words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
		stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
		if method == "mean":
			vectors.append(stacked_vectors.mean(axis=0))
		elif method == "max":
			vectors.append(stacked_vectors.max(axis=0))
		else:
			raise Error("method argument is invalid")
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None,
		id_to_row_index_in_matrix, 
		id_to_col_index_in_matrix, 
		row_index_in_matrix_to_id, 
		col_index_in_matrix_to_id, 
		matrix))




def pairwise_rectangular_bert(model, tokenizer, ids_to_texts_1, ids_to_texts_2, metric, method, layers):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		inferred_vector = _infer_document_vector_from_bert(model, tokenizer, description, method, layers)
		vectors.append(inferred_vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		inferred_vector = _infer_document_vector_from_bert(model, tokenizer, description, method, layers)
		vectors.append(inferred_vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None, 
		id_to_row_index_in_matrix, 
		id_to_col_index_in_matrix, 
		row_index_in_matrix_to_id, 
		col_index_in_matrix_to_id, 
		matrix))








def pairwise_rectangular_ngrams(ids_to_texts_1, ids_to_texts_2, metric, tfidf=False, **kwargs):
	"""
	docstring
	"""
	descriptions = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0
	for identifier,description in ids_to_texts_1.items():
		descriptions.append(description)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		descriptions.append(description)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors,vectorizer = strings_to_vectors(*descriptions, tfidf=tfidf, **kwargs)
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]

	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)

	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, row_id_to_vector_dict, col_id_to_vector_dict, vectorizer, 
		id_to_row_index_in_matrix, 
		id_to_col_index_in_matrix, 
		row_index_in_matrix_to_id, 
		col_index_in_matrix_to_id, 
		matrix))




def pairwise_rectangular_annotations(ids_to_annotations_1, ids_to_annotations_2, ontology, metric, tfidf=False, **kwargs):
	"""
	docstring
	"""
	joined_term_strings = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0
	for identifier,term_list in ids_to_annotations_1.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,term_list in ids_to_annotations_2.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	# Find all the pairwise values for the distance matrix.
	all_vectors,vectorizer = strings_to_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	row_vectors = all_vectors[:len(ids_to_annotations_1)]
	col_vectors = all_vectors[len(ids_to_annotations_1):]

	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)
	
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, row_id_to_vector_dict, col_id_to_vector_dict, vectorizer, 
		id_to_row_index_in_matrix, 
		id_to_col_index_in_matrix, 
		row_index_in_matrix_to_id, 
		col_index_in_matrix_to_id, 
		matrix))































def elemwise_list_doc2vec(model, text_list_1, text_list_2, metric_function):
	"""
	docstring
	"""
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = [model.infer_vector(description.lower().split()) for description in descriptions]
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)




def elemwise_list_word2vec(model, text_list_1, text_list_2, metric_function, method="mean"):
	"""
	docstring
	"""
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = []
	for description in descriptions:
		words = description.lower().split()
		words_in_model_vocab = [word for word in words if word in model.wv.vocab]
		if len(words_in_model_vocab) == 0:
			words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
		stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
		if method == "mean":
			all_vectors.append(stacked_vectors.mean(axis=0))
		elif method == "max":
			all_vectors.append(stacked_vectors.max(axis=0))
		else:
			raise Error("method argument is invalid")

	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)






def elemwise_list_bert(model, tokenizer, text_list_1, text_list_2, metric_function, method, layers):
	"""
	docstring
	"""
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = [_infer_document_vector_from_bert(model, tokenizer, description, method, layers) for description in descriptions]
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)




def elemwise_list_ngrams(text_list_1, text_list_2, metric_function, tfidf=False, **kwargs):
	"""
	docstring	
	"""
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors,vectorizer = strings_to_vectors(*descriptions, **kwargs)
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)





def elemwise_list_annotations(annotations_list_1, annotations_list_2, ontology, metric_function, tfidf=False, **kwargs):
	"""
	docstring	
	"""
	joined_term_strings = []
	all_annotations_lists = annotations_list_1.extend(annotations_list_2)
	for term_list in all_annotations_lists:
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)

	all_vectors,vectorizer = strings_to_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)











































############## Methods for manipulating, combining, and checking edgelists ################






def merge_edgelists(dfs_dict, default_value=None):	
	""" 
	Takes a dictionary mapping between names and {from,to,value} formatted dataframes and
	returns a single dataframe with the same nodes listed but where there is now one value
	column for each dataframe provided, with the name of the column being the corresponding
	name.

	Args:
		dfs_dict (dict): Mapping between strings (names) and pandas.DataFrame objects.
		default_value (None, optional): A value to be inserted where none is present. 
	
	Returns:
		TYPE: Description
	"""

	# Very slow verification step to standardize the dataframes that specify the edge lists.
	# This should only be run if not sure whether or not the methods that generate the edge
	# lists are returning the edges in the exact same format (both i,j and j,i pairs present).
	# dfs_dict = {name:_standardize_edgelist(df) for name,df in dfs_dict.items()}
	# _verify_dfs_are_consistent(*dfs_dict.values())

	for name,df in dfs_dict.items():
		df.rename(columns={"value":name}, inplace=True)
	merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on=["from","to"], how="outer"), dfs_dict.values())
	if not default_value == None:
		merged_df.fillna(default_value, inplace=True)
	return(merged_df)




def make_undirected(df):
	# The dataframe passed in must be in the form {from, to, [other...]}.
	# Convert the undirected edgelist where an edge (j,i) is always implied by an edge (i,j) to a directed edgelist where
	# both the (i,j) and (j,i) edges are explicity present in the dataframe. This is done so that we can make us of the
	# groupby function to obtain all groups that contain all edges between some given node and everything its mapped to 
	# by just grouping base on one of the columns specifying a node. This is easier than using a multi-indexed dataframe.
	other_columns = df.columns[2:]
	flipped_edges = df[flatten(["to","from",other_columns])]      # Create the flipped duplicate dataframe.
	flipped_edges.columns = flatten(["from","to",other_columns])  # Rename the columns so it will stack correctly
	df = pd.concat([df, flipped_edges])
	df.drop_duplicates(keep="first", inplace=True)
	return(df)
	



def remove_self_loops(df):
	return(df[df["from"] != df["to"]])






def standardize_edgelist(df):
	"""
	This function is meant to produce a undirect edge list that contains both the (i,j)
	and (j,i) edges for the entire graph for every possible combination of nodes in the
	graph. Note that this function is supposed to be included so that the methods that
	return the edges lists are not forced to always return complete edge lists. If they
	don't, then when merging edgelists using merge_edgelists, there can be problems when
	doing the outer join, because if only (i,j) is specified then (j,i) will be created 
	but will be NA, which will make it incorrectly seem that that method did not return
	a specific edge value for that pairing.

	Not currently using this method however, because it is slow when the dataset gets
	very large. Currently the methods that generate the edgelist do all pairwise 
	combinations because this is faster, and then the outer join works without 
	introducing incorrectly placed NAs.

	Args:
		df (TYPE): Description
	
	Returns:
		TYPE: Description
	"""

	# Make the edgelist undirected.
	graph = nx.from_pandas_edgelist(df, source="from", target="to", edge_attr=["value"])
	graph = graph.to_undirected()
	graph = graph.to_directed()
	df = nx.to_pandas_edgelist(graph, source="from", target="to")

	# Make sure the nodes are listed in ascending order. If making the edgelist
	# non-redudant instead of redundant.
	# See https://stackoverflow.com/a/45505141
	#cond = df["from"] > df["to"]
	#df.loc[cond, ["from","to"]] = df.loc[cond, ["to","from"]].values	
	return(df)


def subset_edgelist_with_ids(df, ids):
	""" 
	Removes all edges from an edgelist that connects two nodes where one or two of the
	nodes are not present in the passed in list of nodes to retain. This results in an
	edge list that specifies a subgraph of the one specified by the original edgelist.

	Args:
		df (pandas.DataFrame): The edge list before subsetting.
		ids (list): A list of the node IDs that are the only ones left after subsetting.
	Returns:
		pandas.DataFrame: The edge list after subsetting.
	"""
	df = df[df["from"].isin(ids) & df["to"].isin(ids)]
	return(df)



def _verify_dfs_are_consistent(*similarity_dfs):
	"""Check that each dataframe specifies the same set of edges.
	Args:
		*similarity_dfs: Any number of dataframe arguments.
	Raises:
		Error: The dataframes were found to not all be describing the same graph.
	"""
	id_sets = [set() for i in range(0,len(similarity_dfs))]
	for i in range(0,len(similarity_dfs)):
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["from"].values)))
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["to"].values)))
	for (s1, s2) in list(itertools.combinations_with_replacement(id_sets, 2)):	
		if not len(s1.difference(s2)) == 0:
			raise ValueError("dataframes specifying networks are not consisent")










