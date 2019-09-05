from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import os
import sys
import glob
import math
import re
import time

from oats.nlp.search import binary_search_rabin_karp








def get_similarity_df_using_fastsemsim(ontology_obo_file, annotated_corpus_tsv_file, object_dict, duration=False):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	based on the annotations of ontology terms to to all the natural language descriptions that
	those object IDs represent. The ontology files used have to be in the obo format currently and
	any terms that are annotated to the natural language (included in the annnotated corpus file)
	but are not found in the ontology are ignored when calculating similarity. This method also
	assumes that there is a single root to the DAG specified in the obo file which has ID "Thing".
	
	Args:
	    ontology_obo_file (str): File specifying the ontology that was used during annotation.
	    annotated_corpus_tsv_file (str): File specifying the annotations that were made.
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	# Intended values for loading the ontology from a generic obo file. Load the ontology.
	start_time = time.perf_counter()
	ontology_file_type = "obo"
	ontology_type = "Ontology"
	ignore_parameters = {}
	ontology = fss.load_ontology(source_file=ontology_obo_file, ontology_type=ontology_type, file_type=ontology_file_type)

	# Parameters for annotation corpus file with descriptions from fastsemsim documentation.
	ac_source_file_type = "plain"
	ac_params = {}
	ac_params['multiple'] = True 	# Set to True if there are many associations per line (the object in the first field is associated to all the objects in the other fields within the same line).
	ac_params['term first'] = False # Set to True if the first field of each row is a term. Set to False if the first field represents an object.
	ac_params['separator'] = "\t" 	# Select the separtor used to divide fields.
	ac = fss.load_ac(ontology, source_file=annotated_corpus_tsv_file, file_type=ac_source_file_type, species=None, ac_descriptor=None, params = ac_params)

	# Create the object for calculating semantic similarity.
	semsim_type='obj'
	semsim_measure='Jaccard'
	mixing_strategy='max'
	ss_util=None
	semsim_do_log=False
	semsim_params={}
	ss = fss.init_semsim(ontology=ontology, ac=ac, semsim_type=semsim_type, semsim_measure=semsim_measure, mixing_strategy=mixing_strategy, ss_util=ss_util, do_log=semsim_do_log, params=semsim_params)
	
	# Fix issue with lineage object in fastsemsim methods.
	ss.util.lineage = {}
	for node in ontology.nodes:
		ss.util.lineage[node] = "Thing"
	
	# Creating the batch semantic similarity object.
	ssbatch = fss.init_batchsemsim(ontology = ontology, ac = ac, semsim=ss)


	# Generate the pairwise calculations (dataframe) for this batch.
	object_list = list(object_dict.keys())
	result = ssbatch.SemSim(query=object_list, query_type="pairwise")
	result.columns = ["from", "to", "similarity"]


	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)












def get_similarity_df_using_doc2vec(doc2vec_model_file, object_dict, duration=False):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vector embeddings inferred for each natural language description using the passed in 
	Doc2Vec model, which could have been newly trained on relevant data or taken as a pretrained
	model. No assumptions are made about the format of the natural language descriptions, so any
	preprocessing or cleaning of the text should be done prior to being provied in the dictionary
	here.
	
	Args:
	    doc2vec_model_file (str): File where the Doc2Vec model to be loaded is stored.
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""
	start_time = time.perf_counter()
	model = gensim.models.Doc2Vec.load(doc2vec_model_file)

	vectors = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		inferred_vector = model.infer_vector(description.lower().split())
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = cosine_similarity(vectors)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)













# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
def get_cosine_sim_matrix(*strs):
	vectors = [t for t in get_count_vectors(*strs)]
	similarity_matrix = cosine_similarity(vectors)
	return similarity_matrix

def get_count_vectors(*strs):
	text = [t for t in strs]
	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()



def get_jaccard_sim_matrix(*strs):
	vectors = [t for t in get_binary_vectors(*strs)]
	dist = DistanceMetric.get_metric("jaccard")
	similarity_matrix = dist.pairwise(vectors)
	similarity_matrix = 1 - similarity_matrix
	return similarity_matrix

def get_binary_vectors(*strs):
	text = [t for t in strs]
	vectorizer = HashingVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()










def get_similarity_df_using_bagofwords(object_dict, duration=False):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vectors to represent each of the natural language descriptions as a bag-of-words. No 
	assumptions are made about the format of the natural language descriptions, so any cleaning
	or preprocessing of the text should be done prior to being provied in the dictionary here.
	
	Args:
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	start_time = time.perf_counter()
	descriptions = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = get_cosine_sim_matrix(*descriptions)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)










def get_similarity_df_using_setofwords(object_dict, duration=False):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vectors to represent each of the natural language descriptions as a set-of-words. No 
	assumptions are made about the format of the natural language descriptions, so any cleaning
	or preprocessing of the text should be done prior to being provied in the dictionary here.
	
	Args:
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	start_time = time.perf_counter()
	descriptions = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = get_jaccard_sim_matrix(*descriptions)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)









def get_unweighted_jaccard_similarity_of_term_lists(ontology, term_id_list_1, term_id_list_2):
	term_id_set_1 = set()
	term_id_set_2 = set()
	for term_id in term_id_list_1:
		term_id_set_1.update(ontology.subclass_dict[term_id])
	for term_id in term_id_list_2:
		term_id_set_2.udpate(ontology.subclass_dict[term_id])
	return(jaccard_similarity(term_id_set_1, term_id_set_2))

def get_unweighted_jaccard_similarity_of_terms(ontology, term_id_1, term_id_2):
	term_id_set_1 = set(ontology.subclass_dict[term_id_1])
	term_id_set_2 = set(ontology.subclass_dict[term_id_2])
	return(jaccard_similarity(term_id_set_1, term_id_set_2))

def jaccard_similarity(set_1, set_2):
	card_intersection = len(set_1.intersection(set_2))
	car_union = len(set_1.union(set_2))
	similarity = float(card_intersection)/float(card_union)
	return(similarity)







def get_similarity_df_using_annotations_unweighted_jaccard(annotations_dict, ontology, duration=False):
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
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.Dataframe: Each row in the dataframe is [first ID, second ID, similarity].
	"""
	
	# annotations_dict maps object IDs to a list of term annotations.
	start_time = time.perf_counter()
	joined_term_strings = []
	identifier_to_index_in_matrix = {}
	for identifier, term_list in annotations_dict.items():

		# Update the term list to reflect all the terms those terms are subclasses of.
		term_list = [ontology.subclass_dict[x] for x in term_list]
		term_list = itertools.chain.from_iterable(term_list)

		# Prepare to create an indexed matrix from that information.
		index_in_matrix = len(joined_term_strings)
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	# Send the list of term lists to the method which produces a pairwise matrix.
	matrix = get_jaccard_sim_matrix(*joined_term_strings)

	# Produce a pandas dataframe that contains that same information.
	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(annotations_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row
	
	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)






def get_similarity_df_using_annotations_weighted_jaccard(annotations_dict, ontology, duration=False):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	based on the annotations of ontology terms to to all the natural language descriptions that
	those object IDs represent. The individually terms found in the intersection and union between
	two objects are weighted based on their information content as specified in the ontology object.
	This function does not make assumptions about whether the annotations include only leaf terms
	or not, subclass and superclass relationships are accounted for here in either case.
	
	Args:
	    annotations_dict (dict): Mapping from object IDs to lists of ontology term IDs.
	    ontology (Ontology): Ontology object with all necessary fields.
	    duration (bool, optional): Set to true to return the runtime for this method in seconds.
	
	Returns:
	    pandas.Dataframe: Each row in the dataframe is [first ID, second ID, similarity].
	"""
	
	# Produce a pandas dataframe to contain the produced results.
	start_time = time.perf_counter()
	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(annotations_dict, 2)):	
		row = [p1, p2, None]
		result.loc[len(result)] = row

	# Update the annotations_dict to reflect subclass relationships in the ontology.
	expanded_annotations_dict = {}
	for identifier, term_list in annotations_dict.items():
		term_list = [ontology.subclass_dict[x] for x in term_list]
		term_list = list(itertools.chain.from_iterable(term_list))
		expanded_annotations_dict[identifier] = term_list

	# Calculate the Jaccard similarity for each pair of identifiers and update the dataframe.
	for row in result.itertuples():
		term_list_1 = expanded_annotations_dict[row[1]]
		term_list_2 = expanded_annotations_dict[row[2]]
		term_set_1 = set(term_list_1)
		term_set_2 = set(term_list_2)
		intersection = term_set_1.intersection(term_set_2)
		union = term_set_1.union(term_set_2)
		intersection_sum = sum([ontology.ic_dict[x] for x in intersection])
		union_sum = sum([ontology.ic_dict[x] for x in union])
		if union_sum == 0.000:
			similarity = 0.000
		else:
			result.loc[row.Index, "similarity"] = float(intersection_sum)/float(union_sum)
	
	# Return the results.
	if (duration):
		end_time = time.perf_counter()
		duration_in_seconds = end_time-start_time
		return(result,duration_in_seconds)
	return(result)




















