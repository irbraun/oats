import itertools
import re
import string
from nltk import sent_tokenize
from collections import defaultdict
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces


from oats.utils.utils import remove_duplicates_retain_order, flatten
from oats.nlp.small import remove_newlines, add_end_character, remove_punctuation









def remove_text_duplicates_retain_order(texts):
	"""
	Remove the duplicates from a list of text strings, where duplicates are defined as two text strings
	that differ only by puncutation, capitalization, or the length of whitespaces. This is useful for 
	not retaining extra text information just because its not perfectly identical to some existing string.
	Duplicates are removed such that the first occurence is retained, and that determines the final
	ordering. The texts that are returned are not processed, and are a subset of the original list of 
	text strings. The strings retained determined which version of that duplicate in terms of punctuation,
	capitalization, and whitespace is retained in the final list.
	
	Args:
	    texts (list of str): A list of arbitrary strings.
	
	Returns:
	    list of str: A subset of the original list, with duplicates as defined above removed.
	"""
	# Create a list of cleaned texts that corresponds to the list of texts passed in.
	filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]
	cleaned_texts = [" ".join(preprocess_string(text, filters)) for text in texts]
	assert len(texts) == len(cleaned_texts)


	# Get a dictionary mapping the cleaned texts to the a list of the original texts that they resulted from.
	cleaned_to_originals = defaultdict(list)
	for cleaned_text,text in zip(cleaned_texts,texts):
		cleaned_to_originals[cleaned_text].append(text)

	# Remove duplicates and retain the order of the list of the cleaned texts.
	cleaned_texts_no_duplicates = remove_duplicates_retain_order(cleaned_texts)

	# Using whatever the first observed instance of original text that resulting in each cleaned text, rebuild the list.
	original_texts_with_same_removals = [cleaned_to_originals[cleaned_text][0] for cleaned_text in cleaned_texts_no_duplicates]
	return(original_texts_with_same_removals)






def concatenate_texts(texts):
	"""
	Combines multiple description strings into a single string. This is different than a simple join with
	whitespace, because it handles additional formatting which is assumed to be necessary for texts that
	are either fragments or full sentences. This includes removing duplicates that differ only by punctuation
	or capitalization, retaining the specific order of the texts, and making sure they are capitalized and 
	punctuated in a standard way that will be parseable by other packages and functions that deal with text.
	
	Args:
	    texts (list of str): A list of arbitrary strings.
	
	Returns:
	    str: The text string that results from concatenating and formatting these text strings.
	"""
	texts = [text.replace(";",".") for text in texts]
	texts = [add_end_character(text.strip()) for text in texts]
	texts = flatten([sent_tokenize(text) for text in texts])
	texts = remove_text_duplicates_retain_order(texts)
	texts = ["{}{}".format(text[0].upper(), text[1:]) for text in texts]
	text = " ".join(texts).strip()
	text = remove_newlines(text)
	return(text)








############### Things useful for working with lists that are represented by text strings #################


def concatenate_with_delim(delim, elements):
	"""
	Concatenates the strings in the passed in list with a specific delimiter and returns
	the resulting string. This is useful when preparing strings that are intended to be
	placed within a table object or delim-separated text file. Any of the input strings
	can themselves already be representing delim-separated lists, and this will be
	accounted for. 
	
	Args:
	    elements (list of str): A list of strings that represent either lists or list elements.
	
	Returns:
	    str: A text string representing a list that is delimited by the provided delimiter.s
	"""
	tokens = [token.split(delim) for token in elements]
	tokens = flatten(tokens)
	tokens = filter(None, tokens)
	tokens = [token.strip() for token in tokens]
	tokens = remove_duplicates_retain_order(tokens)
	joined = delim.join(tokens).strip()
	joined = remove_newlines(joined)
	return(joined)






def subtract_string_lists(delim, string_list_1, string_list_2):
	"""
	Treats the two input strings as lists that are delimted by the provided delimiter, and
	then returns a new delimited string list that represents the results of the operation
	for treating each list as a set and substracting the second set from the first set.
	
	Args:
	    delim (str): A delimiter for parsing the strings that represent lists.
	    string_list_1 (str): A string that represents a list.
	    string_list_2 (str): A string that represents a list.
	
	Returns:
	    TYPE: A string that represents the list resulting from the operation.
	"""
	tokens1 = string_list_1.split(delim)
	tokens2 = string_list_2.split(delim)
	tokens1 = filter(None, tokens1)
	tokens2 = filter(None, tokens2)
	tokens1 = [token.strip() for token in tokens1]
	tokens2 = [token.strip() for token in tokens2]
	tokens1 = [token for token in tokens1 if token not in tokens2]
	joined = delim.join(tokens1).strip()
	joined = remove_newlines(joined)
	return(joined)








def replace_delimiter(text, old_delim, new_delim):
	"""
	Takes a string that uses one delimiter to represent a list, and returns a new string
	that represents a list using a different delimiter.
	
	Args:
	    text (str): A string that is representing a list using the old delimiter.

	    old_delim (str): Any arbitrary string.
	    
	    new_delim (str): Any arbitrary string.
	
	Returns:
	    str: A string representing a list using the new delimiter.

	"""
	tokens = text.split(old_delim)
	tokens = [token.strip() for token in tokens]
	tokens = [token for token in tokens if not token==""]
	joined = new_delim.join(tokens)
	joined = remove_newlines(joined)
	return(joined) 
























