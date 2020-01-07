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


















def pairwise_doc2vec_onegroup(model, object_dict, metric):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vector embeddings inferred for each natural language description using the passed in 
	Doc2Vec model, which could have been newly trained on relevant data or taken as a pretrained
	model. No assumptions are made about the format of the natural language descriptions, so any
	preprocessing or cleaning of the text should be done prior to being provied in the dictionary
	here.
	
	Args:
		model (gensim.models.doc2vec): An already loaded DocVec model from a file or training.
		object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	
	Returns:
		pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""
	

	# Infer a vector for each unique description, and make the vectors map to indices in a matrix.
	# Remember a mapping between the ID for each object and the positition of the vector for its 
	# description in the matrix so that the similarity values can be recovered later given a pair 
	# of IDs for two objects.
	vectors = []
	index_in_matrix_to_id = {}
	for identifier,description in object_dict.items():
		inferred_vector = model.infer_vector(description.lower().split())
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
	

	# Apply some similarity metric over the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, None))




def pairwise_bert_onegroup(model, tokenizer, object_dict, metric, method, layers):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vector embeddings inferred for each natural language description using the passed in 
	BERT model, which could have been newly trained on relevant data or taken as a pretrained
	model. No assumptions are made about the format of the natural language descriptions, so any
	preprocessing or cleaning of the text should be done prior to being provied in the dictionary
	here.
	"""

	# Infer a vector for each unique description, and make the vectors map to indices in a matrix.
	# Remember a mapping between the ID for each object and the positition of the vector for its 
	# description in the matrix so that the similarity values can be recovered later given a pair 
	# of IDs for two objects.
	vectors = []
	index_in_matrix_to_id = {}
	for identifier,description in object_dict.items():
		inferred_vector = _infer_document_vector_from_bert(model, tokenizer, description, method, layers)
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier

	# Apply some similarity metric over the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, None))



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
	    model (TYPE): Description
	    tokenizer (TYPE): Description
	    description (TYPE): Description
	    method (TYPE, optional): Description
	    layers (TYPE, optional): Description
	
	Returns:
	    TYPE: Description
	
	Raises:
	    Error: The method argument has to be either 'concat' or 'sum'.
	"""
	sentences = sent_tokenize(description)
	token_vecs_cat = []
	token_vecs_sum = []

	# Using the last four layers either concatenated or summed as the embedding.
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

	# Average the vectors obtained for each token across all the sentences.
	if method == "concat":
		embedding = np.mean(np.array(token_vecs_cat),axis=0)
	elif method == "sum":
		embedding = np.mean(np.array(token_vecs_sum),axis=0)
	else:
		raise Error("method argument is invalid")
	return(embedding)















	








def pairwise_word2vec_onegroup(model, object_dict, metric, method="mean"):
	"""	
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vector embeddings inferred for each natural language description using the passed in 
	Word2Vec model, which could have been newly trained on relevant data or taken as a pretrained
	model. No assumptions are made about the format of the natural language descriptions, so any
	preprocessing or cleaning of the text should be done prior to being provied in the dictionary
	here. Note that if no words in a description are in the model vocabulary, then a random word 
	will be selected to represent the text. This avoids using one default value which will force 
	all these descriptions to cluster, and prevents error being raised due to no vector appearing. 
	This should rarely or never happen as long as the text has been preprocessed into reasonable
	tokens.

	Args:
		model (TYPE): Description
		object_dict (TYPE): Description
		metric (TYPE): Description
		method (str, optional): Should the word embeddings be combined with mean or max.
	
	Returns:
		pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""
	vectors = []
	index_in_matrix_to_id = {}
	for identifier,description in object_dict.items():
		words = description.split()
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

	# Apply some similarity metric over the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	return(PairwiseGraph(edgelist, id_to_vector_dict, None, None, None, None))















def pairwise_ngrams_onegroup(object_dict, metric, tfidf=False, **kwargs):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vectors to represent each of the natural language descriptions as a set-of-words. No 
	assumptions are made about the format of the natural language descriptions, so any cleaning
	or preprocessing of the text should be done prior to being provied in the dictionary here.
	
	Args:
		object_dict (dict): Mapping between object IDs and the natural language descriptions. 
		**kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer()
	
	Returns:
		pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	# Map descriptions to coordinates in a matrix.
	descriptions = []
	index_in_matrix_to_id = {}
	for identifier,description in object_dict.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		index_in_matrix_to_id[index_in_matrix] = identifier


	# Find all the pairwise values for the similiarity matrix.
	vectors,vectorizer = strings_to_vectors(*descriptions, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	return(PairwiseGraph(edgelist, id_to_vector_dict, vectorizer, None, None, None))





def pairwise_annotations_onegroup(annotations_dict, ontology, metric, tfidf=False, **kwargs):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	based on the annotations of ontology terms to to all the natural language descriptions that
	those object IDs represent. The individually terms found in the intersection and union between
	two objects are all weighted equally. This function does not make assumptions about whether 
	the annotations include only leaf terms or not, subclass and superclass relationships are 
	accounted for here in either case.
	
	Args:
		annotations_dict (dict): Mapping from object IDs to lists of ontology term IDs.
		ontology (Ontology): Ontology object with all necessary fields.
		**kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer()
	
	Returns:
		pandas.Dataframe: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	# Relevant page for why itertools.chain.from_iterable() can't be used here to flatten the nested string list.
	# https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python

	joined_term_strings = []
	index_in_matrix_to_id = {}
	for identifier,term_list in annotations_dict.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		index_in_matrix = len(joined_term_strings)
		joined_term_strings.append(joined_term_string)
		index_in_matrix_to_id[index_in_matrix] = identifier


	# Find all the pairwise values for the similiarity matrix.
	vectors,vectorizer = strings_to_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	return(PairwiseGraph(edgelist, id_to_vector_dict, vectorizer, None, None, None))










def pairwise_doc2vec_twogroup(model, object_dict_1, object_dict_2, metric):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}

	row_in_matrix = 0	
	for identifier,description in object_dict_1.items():
		inferred_vector = model.infer_vector(description.lower().split())
		vectors.append(inferred_vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in object_dict_2.items():
		inferred_vector = model.infer_vector(description.lower().split())
		vectors.append(inferred_vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(object_dict_1)]
	col_vectors = all_vectors[len(object_dict_1):]
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None, None))







def pairwise_ngrams_twogroup(object_dict_1, object_dict_2, metric, **kwargs):
	"""
	Generate a dataframe that specifies a list of all pairwise edges between nodes
	in the first group and nodes in the second group, based on the similarity 
	between them as assessed by the cosine similarity between their bag-of-words
	representations.
	
	Args:
		object_dict_1 (TYPE): Description
		object_dict_2 (TYPE): Description
		**kwargs: Description
	
	Returns:
		TYPE: Description
	"""
	descriptions = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}

	row_in_matrix = 0
	for identifier,description in object_dict_1.items():
		descriptions.append(description)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in object_dict_2.items():
		descriptions.append(description)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		col_in_matrix = col_in_matrix+1

	all_vectors,vectorizer = strings_to_count_vectors(*descriptions, **kwargs)
	row_vectors = all_vectors[:len(object_dict_1)]
	col_vectors = all_vectors[len(object_dict_1):]
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None, None))







def pairwise_annotations_twogroup(annotations_dict_1, annotations_dict_2, metric):
	"""
	docstring
	"""
	joined_term_strings = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}

	row_in_matrix = 0
	for identifier,term_list in annotations_dict.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		index_in_matrix_to_id[row_in_matrix] = identifier
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,term_list in annotations_dict.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		index_in_matrix_to_id[col_in_matrix] = identifier
		col_in_matrix = col_in_matrix+1


	# Find all the pairwise values for the similiarity matrix.
	all_vectors = strings_to_count_vectors(*joined_term_strings)
	row_vectors = all_vectors[:len(object_dict_1)]
	col_vectors = all_vectors[len(object_dict_1):]
	matrix = cdist(vectors,metric)
	edgelist = rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	return(PairwiseGraph(edgelist, None, None, None, None, None))



























def elemwise_ngrams_twogroup(object_list_1, object_list_2, metric_function, **kwargs):
	""" Takes two lists, zips them, and returns a list of the element-wise distances.
		Has to take a distance function like scipy.spatial.distance.jaccard instead of the name.
	"""
	descriptions = []
	descriptions.extend(object_list_1)
	descriptions.extend(object_list_2)
	all_vectors,vectorizer = strings_to_vectors(*descriptions, **kwargs)
	list_1_vectors = all_vectors[:len(object_list_1)]
	list_2_vectors = all_vectors[len(object_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)


def elemwise_doc2vec_twogroup(model, object_list_1, object_list_2, metric_function):
	""" Takes two lists, zips them, and returns a list of the element-wise distances.
		Has to take a distance function like scipy.spatial.distance.jaccard instead of the name.
	"""
	descriptions = []
	descriptions.extend(object_list_1)
	descriptions.extend(object_list_2)
	all_vectors = [model.infer_vector(description.lower().split()) for description in descriptions]
	list_1_vectors = all_vectors[:len(object_list_1)]
	list_2_vectors = all_vectors[len(object_list_1):]
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	return(distances_list)

def elemwise_word2vec_twogroup():
	return(None)


def elemwise_bert_twogroup():
	return(None)


def elemwise_annotations_twogroup():
	return(None)










































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










