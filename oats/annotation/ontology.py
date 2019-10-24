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






class Ontology:


	def __init__(self, ontology_obo_file):

		self.pronto_ontology_obj = pronto.Ontology(ontology_obo_file)
		self.subclass_dict = self._get_subclass_dict()
		self.ic_dict = self._get_graph_based_ic_dictionary()

		forward_term_dict, reverse_term_dict = self._get_term_dictionaries()
		self.forward_term_dict = forward_term_dict
		self.reverse_term_dict = reverse_term_dict






	def _get_subclass_dict(self):
		"""
		Produces a mapping between ontology term IDs and a list of other IDs which include
		the key ID and all the IDs of every ontology term that that term is a subcless of.
		This means that this can be used to obtain the explicity list of terms that a single
		term has the "is-a" or "part-of" relationship to, for example, in order to calculate
		similarity between ontology terms or sets of them in annotations made.
		
		Returns:
		    dict: The dictionary mapping ontology term IDs to a list of ontology term IDs.
		"""
		subclass_dict = {}
		for term in self.pronto_ontology_obj:
			all_terms = set([x.id for x in term.rparents()])
			all_terms.add(term.id)
			subclass_dict[term.id] = list(all_terms)
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

		# TODO go from counts to ic.

		return(ic_dict)








	def _get_graph_based_ic_dictionary(self):
		"""
		Create a dictionary of information content value for each term in the ontology.
		The equation used for information content here is based on the depth of the term
		which is multiplied by the term [1 - log(descendants)/log(total)]. This works so
		that information content is proportional to depth (increases as terms get more
		specific), but if the number of descendants is very high that value is decreased.

		Returns:
		    dict: Mapping between term IDs and information content values.
		"""

		# TODO depth is not the same thing as number of recursive parents, fix this.
		# TODO find the literature reference or presentation where this equation is from.

		ic_dict = {}
		num_terms_in_ontology = len(self.pronto_ontology_obj)
		for term in self.pronto_ontology_obj:
			depth = len(term.rparents())
			num_descendants = len(term.rchildren())
			ic_value = float(depth)*(1-(math.log(num_descendants+1)/math.log(num_terms_in_ontology)))
			ic_dict[term.id] = ic_value
		return(ic_dict)







	def get_tokens(self):
		"""
		Returns a list of tokens that appear in the labels and synonym strings of this ontology.
		Returns:
		    TYPE: Description
		"""
		labels_and_synonyms = list(itertools.chain.from_iterable(list(self.forward_term_dict.values())))
		tokens = set(list(itertools.chain.from_iterable([word_tokenize(x) for x in labels_and_synonyms])))
		return(list(tokens))

	def get_vocabulary(self):
		""" 
		Returns a dictionary mapping tokens to integers that can be used as a vocabulary. The reason
		this is useful is because the integers (values) are always from 0 to n so that they can be 
		used to map each token to a particular position within a vector, which is these vocabulary
		dictionaries can be used when speciyfing what the token-based vector of some text should be
		formatted like.
		Returns:
		    TYPE: Description
		"""
		labels_and_synonyms = list(itertools.chain.from_iterable(list(self.forward_term_dict.values())))
		tokens = set(list(itertools.chain.from_iterable([word_tokenize(x) for x in labels_and_synonyms])))
		vocabulary = {token:i for i,token in enumerate(list(tokens))}
		return(vocabulary)

	def get_label_from_id(self, term_id):
		try:
			return(self.pronto_ontology_obj[term_id].name)
		except:
			raise KeyError("this identifier matches no terms in the ontology")






























