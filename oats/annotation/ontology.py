from collections import defaultdict
from nltk.tokenize import word_tokenize
import itertools
import pronto
import math
import re

from oats.utils.utils import flatten










class Ontology(pronto.Ontology):


	"""A wrapper class for pronto.ontology.Ontology to provided some extra functions that may be 
	useful for natural language processing problems. Note that the inherited attributes and methods
	aren't documented here, only the additional ones added for this derived class.

	
	Attributes:
	    term_to_tokens (dict of str:list of str): Mapping between ontology term IDs and lists of words that are related to those terms.
	
	    token_to_terms (dict of str:list of str): Mapping between words and the lists of ontology term IDs that are related to those words.
	
	"""
	




	def __init__(self, path):
		"""
		Args:
		    path (str): Path for the .obo file of the ontology to build this object from.
		"""

		# Run the parent class constructor. 
		super(Ontology, self).__init__(path)
		
		# Generate all the data structures that are accessible from instances of the ontology class.
		forward_term_dict, reverse_term_dict = self._get_term_dictionaries()
		self.term_to_tokens = forward_term_dict
		self.token_to_terms = reverse_term_dict

		# Create dictionaries that are used by some of the internal methods for this class.
		self._inherited_dict = self._get_inherited_dict()
		self._depth_dict = self._get_term_depth_dictionary()
		self._graph_based_ic_dict = self._get_graph_based_ic_dictionary()
		





	# Returns a list of tokens that appear in the labels and synonym strings of this ontology.
	# This could be useful for generating a custom vocabulary based on which words are present
	# in descriptions, labels, and synonyms for terms that are in this ontology. Note that this
	# is different than finding the complete set of terms from the vocabulary or their labels,
	# because this splits labels and synonyms that contain multiple words into seperate tokens 
	# at the description of the tokenizer function. Additional processing might be needed to turn
	# this set of tokens into a useful vocabulary. 
	def tokens(self):
		"""
		Get a list of the tokens or words that appear in this ontology. 
		This is intented to be useful for treating the ontology as a vocabulary source.
		
		Returns:
		    list of str: Lists of words in the set of all words present in all term labels and synonyms in this ontology.
		"""
		labels_and_synonyms = list(itertools.chain.from_iterable(list(self.term_to_tokens.values())))
		tokens = set(list(itertools.chain.from_iterable([word_tokenize(x) for x in labels_and_synonyms])))
		return(list(tokens))





	def depth(self, term_id):
		"""
		Given an ontology term ID, return the depth of that term in the hierarchial ontology graph.
		The depth provided is an integer that indicates the shortest possible path from that term to a root term.
		
		Args:
		    term_id (str): The ID for a particular ontology term.
		
		Returns:
		    int: The depth of the term.
		"""
		return(self._depth_dict.get(term_id, None))





	def ic(self, term_id):
		"""
		Given an ontology term ID, return the information content of that term from the structure of the 
		hierarchical ontology graph. This information content value takes into account the depth of the
		term in the graph, as well as what proportion of the total graph is a descendent of this term.
		The equation used for information content here is based on the depth of the term
		which is multiplied by the term [1 - log(descendants+1)/log(total)]. This works so
		that information content is proportional to depth (increases as terms get more
		specific), but if the number of descendants is very high that value is decreased.
		This is an alternative to calculating information content directly from the ontology graph rather 
		than using the frequencies of terms appearing in data. This is useful when no such resource is 
		available.
		
		Args:
		    term_id (str): The ID for a particular ontology term.
		
		Returns:
		    float: The information content of the term.
		"""
		return(self._graph_based_ic_dict.get(term_id, None))




	def inherited(self, term_ids):
		"""
		Given an ontology term ID, return a list of the term IDs for all the terms that are inherited
		by this particular term, including the term itself. The list is prepopulated using the pronto 
		superclases method. The only difference is that a list of term ID strings is provided instead 
		of a generator of term objects, which was useful for other methods in this class. This also
		accepts a list of one or more term IDs in which the union of the terms inherited by all terms
		in the list are returned, including every term in the passed in list.
		
		Args:
		    term_ids (list of str or str): The ID for a particular ontology, or a list of ID(s).
		
		Returns:
		    TYPE: Description
		"""

		if isinstance(term_ids, str):
			term_ids = [term_ids]
		inherited_ids = []
		for term_id in term_ids:
			inherited_ids.extend(self._inherited_dict.get(term_id, [term_id]))
		return(list(set(inherited_ids)))
		















	def _get_inherited_dict(self):
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
		for term in self.terms():
			subclass_dict[term.id] = [t.id for t in self[term.id].superclasses(with_self=True)]
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
		for term in self.terms():
			if (term.name is not None) and ("obsolete" not in term.name):  
				words = [term.name]
				words.extend([x.description for x in list(term.synonyms)])	# Add all the synonyms
				words = [re.sub(r" ?\([^)]+\)", "", x) for x in words]		# Replace parenthetical text.
				forward_dict[term.id] = words
				for word in words:
					reverse_dict[word].append(term.id)
		return(forward_dict, reverse_dict)




	






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
		num_terms_in_ontology = len(self)
		for term in self.terms():
			depth = self._depth_dict[term.id]
			num_descendants = len(list(term.subclasses(with_self=False)))
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



		# Find the root term(s) of the ontology.
		root_term_ids = []
		for term in self.terms():
			# Check if this term has no inherited terms (is a root), discounting terms that are obsolete.
			inherited_terms = [t for t in term.superclasses(with_self=False)]
			if (len(inherited_terms)==0) and (term.name is not None) and ("obsolete" not in term.name):
				root_term_ids.append(term.id)
				
		# Find the depths of all terms in the ontology below those terms.
		depths = {i:0 for i in root_term_ids}
		depth = 1
		done = False
		while not done:
			
			# Add all the terms immediately below 
			before = len(depths)
			new_terms = []
			for old_term_id in [i for i in depths.keys() if depths[i] == depth-1]:
				for new_term_id in [t.id for t in self[old_term_id].subclasses(with_self=False,distance=1)]:
					if new_term_id not in depths:
						depths[new_term_id] = depth
			
			# Increment the depth and see if any new terms were added to the distance dictionary during this pass.
			depth = depth + 1
			after = len(depths)
			if before == after:
				done = True
				
		# Add any other remaining terms to the dictionary with a depth of 0 indicating minimal specificity.
		for term in self.terms():
			if term.id not in depths:
				depths[term.id] = 0
		
		# Return the dictionary mapping term IDs to their depth in the hierarchy.
		return(depths)













	def similarity_jaccard(self, term_id_list_1, term_id_list_2, inherited=True):
		"""
		Find the similarity between two lists of ontology terms, by finding the Jaccard
		similarity between the two sets of all the terms that are inherited by each of
		the terms present in each list. 
		
		Args:
			term_id_list_1 (list of str): A list of ontology term IDs.

			term_id_list_2 (list of str): A list of ontology term IDs.

			inherited (bool, optional): Should other terms inherited by these terms be accounted for.
		
		Returns:
			float: The jaccard similarity between the two lists of terms.
		"""
		if inherited:
			term_id_set_1 = set(term_id_list_1)
			term_id_set_2 = set(term_id_list_2)
		else:
			inherited_term_list_1 = flatten(self._inherited_dict[term_id] for term_id in term_id_list_1)
			inherited_term_list_2 = flatten(self._inherited_dict[term_id] for term_id in term_id_list_2)
			term_id_set_1 = set(inherited_term_list_1)
			term_id_set_2 = set(inherited_term_list_2)

		intersection = term_id_set_1.intersection(term_id_set_2)
		union = term_id_set_1.union(term_id_set_2)
		return(len(intersection)/len(union))






	def similarity_ic(self, term_id_list_1, term_id_list_2, inherited=True):
		"""
		Find the similarity between two lists of ontology terms, by finding the information 
		content of the most specific term that is shared by the sets of all terms inherited
		by all terms in each list. In this case, the most specific term is the term with
		maximum information content.

		Args:
			term_id_list_1 (list of str): A list of ontology term IDs.

			term_id_list_2 (list of str): A list of ontology term IDs.

			inherited (bool, optional): Should other terms inherited by these terms be accounted for.
		
		Returns:
			float: The maximum information content of any common ancestor between the two term lists.
		"""
		if inherited:
			term_id_set_1 = set(term_id_list_1)
			term_id_set_2 = set(term_id_list_2)
		else:
			inherited_term_list_1 = flatten(self._inherited_dict[term_id] for term_id in term_id_list_1)
			inherited_term_list_2 = flatten(self._inherited_dict[term_id] for term_id in term_id_list_2)
			term_id_set_1 = set(inherited_term_list_1)
			term_id_set_2 = set(inherited_term_list_2)

		intersection = list(term_id_set_1.intersection(term_id_set_2))
		intersection_ic_values = [self._graph_based_ic_dict[term_id] for term_id in intersection]
		if len(intersection_ic_values) == 0:
			return(0.000)
		return(max(intersection_ic_values))











































