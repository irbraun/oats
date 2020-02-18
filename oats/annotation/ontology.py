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
from nltk.tokenize import word_tokenize

from oats.nlp.search import binary_search_rabin_karp
from oats.utils.utils import flatten






class Ontology:

	"""A wrapper class for pronto.Ontology to provide some extra NLP-centric functions.
	
	Attributes:
	    forward_term_dict (dict of str:list of str): Mapping between ontology term IDs and lists of words that are related to those terms.

	    ic_dict (dict of str:float): Mapping between ontology term IDs and the information content of that term in the context of the ontology graph.
	    
	    pronto_ontology_obj (pronto.Ontology): The ontology object from the pronto package that this object is wrapping.
	    
	    reverse_term_dict (dict of str:list of str): Mapping between words and the lists of ontology term IDs that are related to those words.
	    
	    subclass_dict (dict of str:list of str): Mapping between ontology term IDs and a list of ontology term IDs of all terms inherited by the key term.
	"""
	




	def __init__(self, ontology_obo_file):
		"""Initiates an object of this class.
		
		Args:
		    ontology_obo_file (str): Path of the .obo file of the ontology to build this object from.

		"""
		
		# Generate all the data structures that are accessible from instances of the ontology class.
		self.pronto_ontology_obj = pronto.Ontology(ontology_obo_file)
		self.subclass_dict = self._get_subclass_dict()
		self.depth_dict = self._get_term_depth_dictionary()
		self.graph_based_ic_dict = self._get_graph_based_ic_dictionary()
		forward_term_dict, reverse_term_dict = self._get_term_dictionaries()
		self.forward_term_dict = forward_term_dict
		self.reverse_term_dict = reverse_term_dict






	# Returns a list of tokens that appear in the labels and synonym strings of this ontology.
	# This could be useful for generating a custom vocabulary based on which words are present
	# in descriptions, labels, and synonyms for terms that are in this ontology. Note that this
	# is different than finding the complete set of terms from the vocabulary or their labels,
	# because this splits labels and synonyms that contain multiple words into seperate tokens 
	# at the description of the tokenizer function. Additional processing might be needed to turn
	# this set of tokens into a useful vocabulary. 
	def get_tokens(self):
		""" Gets the tokens (words) which appear in this ontology.

		Returns:
		    list of str: Lists of words in the set of words present in all term labels and synonyms in this ontology.
		"""
		labels_and_synonyms = list(itertools.chain.from_iterable(list(self.forward_term_dict.values())))
		tokens = set(list(itertools.chain.from_iterable([word_tokenize(x) for x in labels_and_synonyms])))
		return(list(tokens))






	def get_label_from_id(self, term_id):
		"""Gets the label corresponding to an ontology term ID.
		
		Args:
		    term_id (str): The ID string for some term.
		
		Returns:
		    str: The label corresponding to that term ID.
		
		Raises:
		    KeyError: This ID does not refer to a term in the ontology.
		"""
		try:
			return(self.pronto_ontology_obj[term_id].name)
		except:
			raise KeyError("this identifier matches no terms in the ontology")










	def _get_inherited_term_ids(self, term_id):
		"""Gets all the terms inherited by a given term.
		
		Args:
		    term_id (TYPE): The ID string for some term.
		
		Returns:
		    TYPE: A list of additional term IDs which are inherited by this term.
		
		Raises:
		    KeyError: This ID does not refer to a term in the ontology.
		"""
		try:
			term = self.pronto_ontology_obj[term_id]
		except:
			raise KeyError("this identifier matches no terms in the ontology")
		inherited_terms = [x.id for x in term.rparents()]
		return(inherited_terms)







	def _get_subclass_dict(self):
		"""
		Produces a mapping between ontology term IDs and a list of other IDs which include
		the key ID and all the IDs of every ontology term that that term is a subclass of.
		This means that this can be used to obtain the explicity list of terms that a single
		term has the "is-a" or "part-of" relationship to, for example, in order to calculate
		similarity between ontology terms or sets of them in annotations made.
		
		Returns:
		    dict: The dictionary mapping ontology term IDs to a list of ontology term IDs.
		"""
		subclass_dict = {}
		for term in self.pronto_ontology_obj:
			subclass_dict[term.id] = self._get_inherited_term_ids(term.id)
		return(subclass_dict)






	def _get_term_dictionaries(self):
		"""
		Produces a mapping between ontology term IDs and a list of the strings which are related
		to them (the name of the term and any synonyms specified in the ontology) which is the
		forward dictionary, and a mapping between strings and all the ontology term IDs that those
		strings were associated with, which is the reverse mapping.
		
		Returns:
		    (dict, dict): The forward and reverse mapping dictionaries.
		"""
		forward_dict = {}
		reverse_dict = defaultdict(list)
		for term in self.pronto_ontology_obj:
			if "obsolete" not in term.name:
				words = [term.name]
				words.extend([x.desc for x in list(term.synonyms)])			# Add all the synonyms
				words = [re.sub(r" ?\([^)]+\)", "", x) for x in words]		# Replace parenthetical text.
				forward_dict[term.id] = words
				for word in words:
					reverse_dict[word].append(term.id)
		return(forward_dict, reverse_dict)




	

	def _get_corpus_based_ic_dictionary_from_annotations(self, annotations_dict):
		"""
		Create a dictionary of information content values for each term in the ontology.
		Use the frequency of term IDs included in any text file (such as an annotation)
		file in order to accomplish this. This method accounts for subclass relationships
		between terms so that if a some term and one of its children are both mentioned
		in the text file, the original term is counted twice and the child is counted once.
		
		Args:
		    annotations_dict (dict): Mapping between identifiers and lists of ontology terms.
		
		Returns:
		    dict: Mapping between term IDs and information content values.
		"""

		# TODO write this
		return(ic_dict)






	def _get_corpus_based_ic_dictionary_from_raw_counts_in_text(self, corpus_filename):
		"""
		Create a dictionary of information content values for each term in the ontology.
		Use the frequency of term IDs included in any text file (such as an annotation)
		file in order to accomplish this. This method accounts for subclass relationships
		between terms so that if a some term and one of its children are both mentioned
		in the text file, the original term is counted twice and the child is counted once.
		
		Args:
		    corpus_filename (str): Path to the file to be used as the corpus.
		
		Returns:
		    dict: Mapping between term IDs and information content values.
		"""

		occurence_count_dict = {}
		corpus = open(corpus_filename, "r").read()
		for term_id in self.subclass_dict.keys():
			count = corpus.count(term_id)
			occurence_count_dict[term_id] = count

		ic_dict = {}	

		# TODO go from counts to information content.
		return(ic_dict)








	def _get_graph_based_ic_dictionary(self):
		"""
		Create a dictionary of information content value for each term in the ontology.
		The equation used for information content here is based on the depth of the term
		which is multiplied by the term [1 - log(descendants+1)/log(total)]. This works so
		that information content is proportional to depth (increases as terms get more
		specific), but if the number of descendants is very high that value is decreased.

		Returns:
		    dict: Mapping between term IDs and information content values.
		"""

		# TODO find the literature reference or presentation where this equation is from.

		ic_dict = {}
		num_terms_in_ontology = len(self.pronto_ontology_obj)
		for term in self.pronto_ontology_obj:
			depth = self.depth_dict[term.id]
			num_descendants = len(term.rchildren())
			ic_value = float(depth)*(1-(math.log(num_descendants+1)/math.log(num_terms_in_ontology)))
			ic_dict[term.id] = ic_value
		return(ic_dict)







	def _get_term_depth_dictionary(self):
		"""
		Create a dictionary of the depths of each term in the ontology. This is used in 
		calculating the graph-based information content of each term. This value is not
		the same as the number of recursive parents for a given term, because a single
		term can have two or more parents in a DAG. In other words, multiple paths can 
		exist from a term to the root of the graph, and the depth of the term should be
		the minimum path length out of all of the possible paths.
		
		Returns:
		    dict of str:int: Mapping between term IDs and their depth in the DAG.
		"""
		depth_dict = {}
		for term in self.pronto_ontology_obj:
			depth_dict[term.id] = self._get_depth_recursive(term, 0)
		return(depth_dict)


	def _get_depth_recursive(self, term, depth):
		if len(term.parents) == 0:
			return(depth)
		else:
			depths = []
			for parent in term.parents:
				depths.append(self._get_depth_recursive(parent, depth+1))
			return(min(depths))







	def jaccard_similarity(self, term_id_list_1, term_id_list_2, inherited=True):
		"""
		Find the similarity between two lists of ontology terms, by finding the Jaccard
		similarity between the two sets of all the terms that are inherited by each of
		the terms present in each list. 
		
		Args:
		    term_id_list_1 (TYPE): Description

		    term_id_list_2 (TYPE): Description

		    inherited (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if inherited:
			term_id_set_1 = set(term_id_list_1)
			term_id_set_2 = set(term_id_list_2)
		else:
			inherited_term_list_1 = flatten(self.subclass_dict[term_id] for term_id in term_id_list_1)
			inherited_term_list_2 = flatten(self.subclass_dict[term_id] for term_id in term_id_list_2)
			inherited_term_list_1.extend(term_id_list_1)
			inherited_term_list_2.extend(term_id_list_2)
			term_id_set_1 = set(inherited_term_list_1)
			term_id_set_2 = set(inherited_term_list_2)

		intersection = term_id_set_1.intersection(term_id_set_2)
		union = term_id_set_1.union(term_id_set_2)
		return(len(intersection)/len(union))




	

	def info_content_similarity(self, term_id_list_1, term_id_list_2, inherited=True):
		"""
		Find the similarity between two lists of ontology terms, by finding the information 
		content of the most specific term that is shared by the sets of all terms inherited
		by all terms in each list. 

		Args:
		    term_id_list_1 (TYPE): Description

		    term_id_list_2 (TYPE): Description

		    inherited (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if inherited:
			term_id_set_1 = set(term_id_list_1)
			term_id_set_2 = set(term_id_list_2)
		else:
			inherited_term_list_1 = flatten(self.subclass_dict[term_id] for term_id in term_id_list_1)
			inherited_term_list_2 = flatten(self.subclass_dict[term_id] for term_id in term_id_list_2)
			inherited_term_list_1.extend(term_id_list_1)
			inherited_term_list_2.extend(term_id_list_2)
			term_id_set_1 = set(inherited_term_list_1)
			term_id_set_2 = set(inherited_term_list_2)

		intersection = list(term_id_set_1.intersection(term_id_set_2))
		intersection_ic_values = [self.graph_based_ic_dict[term_id] for term_id in intersection]
		if len(intersection_ic_values) == 0:
			return(0.000)
		return(max(intersection_ic_values))











































