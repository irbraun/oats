from rapidfuzz import fuzz
from rapidfuzz import process









def binary_robinkarp_match(pat, txt, q=1193):
	"""
	Searches for exact matches to a pattern in a longer string. 
	Adapted from implementation by Bhavya Jain from https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/.
	The Robin Karp algorithm is a fast algorithm for finding exact matches between a pattern and a longer string, so a match
	is only considered real if it matches character for character. 
	
	Args:
		pat (str): The shorter text to search for.
	
		txt (str): The larger text to search within.
	
		q (int, optional): A prime number used for hashing.
	
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




def any_rabinkarp_matches(patterns, text, q=1193):
	"""
	Return true if any pattern from a list of patterns is in the text, false else. 
	The Robin Karp algorithm is a fast algorithm for finding exact matches between a 
	pattern and a longer string, so a match is only considered real if it matches 
	character for character. 
	
	Args:
		patterns (str): The shorter text to search for.
	
		text (str): The larger text to search within.
	
		q (int, optional): A prime number used for hashing.
	
	Returns:
		boolean: True if any of the patterns were found, false if none were.
	"""
	for pattern in patterns:
		if binary_robinkarp_match(pattern, text, q):
			return(True)
	return(False)



def all_rabinkarp_matches(patterns, text, q=1193):
	"""
	Returns the sublist of patterns that appear in the text.
	The Robin Karp algorithm is a fast algorithm for finding exact matches between a 
	pattern and a longer string, so a match is only considered real if it matches 
	character for character. 
	
	Args:
		patterns (str): The shorter text to search for.
	
		text (str): The larger text to search within.
	
		q (int, optional): A prime number used for hashing.
	
	Returns:
		list: A sublist of the patterns argument containing only the patterns that were found.
	"""
	patterns_found = []
	for pattern in patterns:
		if binary_robinkarp_match(pattern, text, q):
			patterns_found.append(pattern)
	return(patterns_found)











def binary_fuzzy_match(pat, txt, threshold, local=1):
	"""
	Searches for fuzzy matches to a pattern in a longer string. A fuzzy match does 
	not necessarily need to be a perfect character for character match between a pattern
	and the larger text string, with a tolerance for mismatches controlled by the 
	threhsold parameter. The underlying metric is Levenshtein distance.
	
	Args:
		pat (str): The shorter text to search for.
		
		txt (str): The larger text to search within.
		
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





def any_fuzzy_matches(patterns, text, threshold, local=1):
	"""
	Return true if any pattern from a list of patterns is in the text, false else.
	A fuzzy match does not necessarily need to be a perfect character for character 
	match between a pattern and the larger text string, with a tolerance for mismatches 
	controlled by the threhsold parameter. The underlying metric is Levenshtein distance.
	
	Args:
		patterns (list): The shorter text strings to search for.
	
		txt (str): The larger text to search within.

		threshold (float): Value between 0 and 1 at which matches are considered positive.

		local (int, optional): Alignment method, 0 for global 1 for local.
	
	Returns:
		list: A sublist of the patterns argument containing only the patterns that were found.
	"""
	for pattern in patterns:
		if binary_fuzzy_match(pattern, text, threshold, local):
			return(True)
	return(False)





def all_fuzzy_matches(patterns, txt, threshold, local=1):
	"""
	Returns the sublist of patterns that appear in the text.
	A fuzzy match does not necessarily need to be a perfect character for character 
	match between a pattern and the larger text string, with a tolerance for mismatches 
	controlled by the threhsold parameter. The underlying metric is Levenshtein distance.
	
	Args:
		patterns (list): The shorter text strings to search for.
	
		txt (str): The larger text to search within.

		threshold (float): Value between 0 and 1 at which matches are considered positive.

		local (int, optional): Alignment method, 0 for global 1 for local.
	
	Returns:
		list: A sublist of the patterns argument containing only the patterns that were found.
	"""
	# The method process.extract() returns a list of tuples where the first
	# item is the pattern string and the second item is the alignment score for 
	# that pattern.
	patterns_found = []
	threshold = threshold*100
	if local==1:
		method = fuzz.partial_ratio
	else:
		method = fuzz.ratio
	# Note that for rapidfuzz==0.3.0 rapidfuzz.proces.extract() matches must exceed the threshold, 
	# not meet it like in rapidfuzz.fuzz.partial_ratio() and rapidfuzz.fuzz.ratio().
	best_matches = process.extract(query=txt, choices=patterns, scorer=method, score_cutoff=threshold)
	patterns_found = [match[0] for match in best_matches]
	return(patterns_found)





















