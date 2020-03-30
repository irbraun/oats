from rapidfuzz import fuzz
from rapidfuzz import process









def binary_search_rabin_karp(pat, txt, q=1193): 
	"""
	Searches for exact matches to a pattern in a longer string (fast). 
	Adapted from implementation by Bhavya Jain from
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




def search_for_any_rabin_karp(patterns, text, q=1193):
	"""Return true if any pattern from a list of patterns is in the text, false else.
	"""
	for pattern in patterns:
		if binary_search_rabin_karp(pattern, text, q):
			return(True)
	return(False)



def search_for_all_rabin_karp(patterns, text, q=1193):
    """Returns the sublist of patterns that appear in the text.
    """
    patterns_found = []
    for pattern in patterns:
    	if binary_search_rabin_karp(pattern, text, q):
    		patterns_found.append(pattern)
    return(patterns_found)

























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



def search_for_any_fuzzy(patterns, text, threshold, local=1):
	"""Return true if any pattern from a list of patterns is in the text, false else.
	"""
	for pattern in patterns:
		if binary_search_fuzzy(pattern, text, threshold, local):
			return(True)
	return(False)



def search_for_all_fuzzy(patterns, txt, threshold, local=1):
	"""
	Searches for occurences of any of the patterns in the longer string (slow).
	The method process.extract() returns a list of tuples where the first
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
	best_matches = process.extract(query=txt, choices=patterns, scorer=method, score_cutoff=threshold)
	patterns_found = [match[0] for match in best_matches]
	return(patterns_found)





















