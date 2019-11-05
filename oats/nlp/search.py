import numpy as np
import string
import itertools
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from oats.utils.utils import remove_duplicates_retain_order






def binary_search_rabin_karp(pat, txt, q=1193): 
	"""
	Searches for exact matches to a pattern in a longer string (fast). 
	Adapted from implementation by Bhavya Jain found at
	https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
	Args:
		pat (str): The shorter text to search for.
		txt (str): The larger text to search in.
		q (int): A prime number that is used for hashing.
	Returns:
		boolean: True if the pattern was found, false is it was not.
	"""
	# Make sure the pattern is smaller than the text.
	if len(pat)>len(txt):
		return(False)
	d = 256				# number of characters in vocabulary
	M = len(pat) 
	N = len(txt) 
	i = 0
	j = 0
	p = 0    			# hash value for pattern 
	t = 0    			# hash value for txt 
	h = 1
	found_indices = []
	for i in range(M-1): 
		h = (h * d)% q 
	for i in range(M): 
		p = (d * p + ord(pat[i]))% q 
		t = (d * t + ord(txt[i]))% q 
	for i in range(N-M + 1): 
		if p == t: 
			for j in range(M): 
				if txt[i + j] != pat[j]: 
					break
			j+= 1
			if j == M: 
				# Pattern found at index i.
				# found_indices.append(i)
				return(True)
		if i < N-M: 
			t = (d*(t-ord(txt[i])*h) + ord(txt[i + M]))% q 
			if t < 0: 
				t = t + q 
	# Pattern was never found.			
	return(False)





def check_text_for_patterns_rabin_karp(patterns, text, q=1193):
	for pattern in patterns:
		if binary_search_rabin_karp(pattern, text, q):
			return(True)
	return(False)








































def binary_search_fuzzy(pat, txt, threshold, local=1):
	"""Searches for fuzzy matches to a pattern in a longer string (slow).
	Args:
		pat (str): The shorter text to search for.
		txt (str): The larger text to search in.
		threshold (int): Value between 0 and 1 at which matches are considered real.
		local (int, optional): Alignment method, 0 for global 1 for local.
	Returns:
		boolean: True if the pattern was found, false if it was not.
	"""
	# Make sure the pattern is smaller than the text.
	if len(pat)>len(txt):
		return(False)
	similarity_score = 0.000
	if local==1:
		similarity_score = fuzz.partial_ratio(pat, txt)
	else:
		similarity_score = fuzz.ratio(pat, txt)
	if similarity_score >= threshold*100:
		return(True)
	return(False)




def occurences_of_patterns_in_text_fuzzy(patterns, txt, threshold, local=1):
	"""
	Searches for occurences of any of the patterns in the longer string (slow).
	The method process.extractBests() returns a list of tuples where the first
	item is the pattern string and the second item is the alignment score for 
	that pattern.
	
	Args:
		patterns (list): The shorter text strings to search for.
		txt (str): The larger text to search in.
		threshold (int): Value between 0 and 1 at which matches are considered real.
		local (int, optional): Alignment method, 0 for global 1 for local.
	Returns:
		list: A sublist of the patterns argument containing only the found strings.
	"""
	patterns_found = []
	threshold = threshold*100
	if local==1:
		method = fuzz.partial_ratio
	else:
		method = fuzz.ratio
	best_matches = process.extractBests(query=txt, choices=patterns, scorer=method, score_cutoff=threshold)
	patterns_found = [match[0] for match in best_matches]
	return(patterns_found)





def occurences_of_pattern_in_texts_fuzzy(pattern, texts, threshold, local=1):
	"""Searches for occurences of a pattern in any of the longer strings.
	Args:
	    pattern (TYPE): Description
	    texts (TYPE): Description
	    threshold (TYPE): Description
	    local (int, optional): Description
	Returns:
	    TYPE: Description
	"""
	matching_texts = []
	for txt in texts:
		if binary_search_fuzzy(pattern, txt, threshold, local=local):
			matching_texts.append(txt)
	return(matchin_texts)





















