from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import numpy as np
import pandas as pd
import random

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









def _strings_to_count_vectors(texts, training_texts=None, **kwargs):
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
	vectorizer = CountVectorizer(**kwargs)
	if training_texts is not None:
		vectorizer.fit(training_texts)
	else:
		vectorizer.fit(texts)
	vectors = vectorizer.transform(texts).toarray()
	return(vectors, vectorizer)





def _strings_to_tfidf_vectors(texts, training_texts=None, **kwargs):
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
	vectorizer = TfidfVectorizer(**kwargs)
	if training_texts is not None:
		vectorizer.fit(training_texts)
	else:
		vectorizer.fit(texts)
	vectors = vectorizer.transform(texts).toarray()
	return(vectors, vectorizer)






















