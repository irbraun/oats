import itertools
import re
import string
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
	texts = remove_text_duplicates_retain_order(texts)
	texts = [add_end_character(text.strip()) for text in texts]
	text = " ".join(texts).strip()
	text = remove_newlines(text)
	return(text)








############### Things useful for converting between strings and lists ################


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








def replace_delimiter_safely(text, old_delim, new_delim):
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








############### General-purpose functions for preprocessing text ################






# def get_clean_text(text):
# 	"""Clean the text by removing punctuation and normalizing case.
	
# 	Args:
# 	    text (str): Any piece of text.
	
# 	Returns:
# 	    str: That text cleaned.
# 	"""
# 	text = remove_punctuation(text)
# 	text = text.lower()
# 	return(text)





# def get_clean_token_list(text):
# 	"""Cleans the text by removing punctuation and normalizing case and then splits on 
# 	whitespace to return a list of tokens.
	
# 	Args:
# 	    text (str): Any piece of text.
	
# 	Returns:
# 	    list of str: A list of tokens.
# 	"""
# 	text = remove_punctuation(text)
# 	token_list = text.lower().split()
# 	return(token_list)





# def remove_punctuation(text):
# 	"""Remove all punctuation from a piece of text.
	
# 	Args:
# 	    text (str): Any piece of text.
	
# 	Returns:
# 	    str: That text without any characters that were punctuation.
	
# 	"""
# 	translator = str.maketrans('', '', string.punctuation)
# 	return(text.translate(translator))






# def remove_newlines(text):
# 	"""Remove all newline characters from a piece of text.
	
# 	Args:
# 	    text (str): Any piece of text.
	
# 	Returns:
# 	    str: That text without newline characters.
# 	"""
# 	text = text.replace("\n", " ")
# 	text = text.replace("\t", " ")
# 	text = text.replace("\r", " ")
# 	return(text)





# def add_end_char(text):
# 	"""Adds a period to the end of the text. This could be useful when concatentating
# 	text while still retaining the sentence or phrase boundaries taken into account by
# 	other processing steps such as part-of-speech analysis. Accounts for text that ends
# 	in periods or semicolons.
	
# 	Args:
# 	    text (str): Any piece of text.
	
# 	Returns:
# 	    str: That text with a period added to the end.
# 	"""
# 	if len(text) > 0:
# 		last_character = text[len(text)-1]
# 		end_tokens = [".", ";"]
# 		if not last_character in end_tokens:
# 			text = text+"."
# 	return(text)










# ############### Things useful for converting between strings and lists ################


# def concatenate_with_bar_delim(*tokens):
# 	"""
# 	Concatenates any number of passed in tokens with a bar character and returns the 
# 	resulting string. This is useful for preparing things like gene names for entry
# 	into a table that could be written to a csv where multiple elements need to go 
# 	into a single column but a standard delimiter like a comma should not be used.
# 	Some of the tokens passed in can strings that are already delimited with a bar 
# 	character. These are treated as seperate elements that should be incorporated 
# 	into the string that is returned, trailing and leading bars are handled.
	
# 	Args:
# 	    *tokens: Any number of arbitrary strings, can include strings already delimited.
	
# 	Returns:
# 	    str: Tokens separated by a bar delimiter.
# 	"""
# 	tokens = [token.split("|") for token in tokens]
# 	tokens = itertools.chain.from_iterable(tokens)
# 	tokens = filter(None, tokens)
# 	tokens = [token.strip() for token in tokens]
# 	tokens = remove_duplicates_retain_order(tokens)
# 	joined = "|".join(tokens).strip()
# 	joined = remove_newlines(joined)
# 	return(joined)





# def remove_occurences_from_bar_delim_lists(dar_delim_string_1, bar_delim_string_2):
# 	"""Removes elements from list 1 that are already in list 2 and returns list 1.
	
# 	Args:
# 	    dar_delim_string_1 (TYPE): Description
# 	    bar_delim_string_2 (TYPE): Description
# 	"""
# 	tokens1 = dar_delim_string_1.split("|")
# 	tokens2 = bar_delim_string_2.split("|")
# 	tokens1 = filter(None, tokens1)
# 	tokens2 = filter(None, tokens2)
# 	tokens1 = [token.strip() for token in tokens1]
# 	tokens2 = [token.strip() for token in tokens2]
# 	tokens1 = [token for token in tokens1 if token not in tokens2]
# 	joined = "|".join(tokens1).strip()
# 	joined = remove_newlines(joined)
# 	return(joined)





# def other_delim_to_bar_delim(string, delim):
# 	"""Convert string delimited with some character (semicolon, comma, etc) to one delimited with bars.
	
# 	Args:
# 	    string (str): The string that is currently delimited by the passed in delimiter.

# 	    delim (str): The delimiter present in the string which should be a single character.
	
# 	Returns:
# 	    str: The string separated by the bar delimiter.
	
# 	Raises:
# 	    ValueError: The current delimiting string should be a single character.
# 	"""
# 	if not len(delim) == 1:
# 		raise ValueError("delimiter {} is not a single character".format(delim))
# 	tokens = string.split(delim)
# 	tokens = [token.strip() for token in tokens]
# 	tokens = [token for token in tokens if not token==""]
# 	joined = "|".join(tokens)
# 	joined = remove_newlines(joined)
# 	return(joined) 

















# ############### More case-specific functions that are sometimes for working with gene names ################



# def remove_enclosing_brackets(string):
# 	"""If brackets enclose the string, this method removes them.
	
# 	Args:
# 	    string (str): Any arbitrary string.
	
# 	Returns:
# 	    str: The same string but with the brackets characters removed if they were enclosing the string. 
# 	"""
# 	if string.startswith("[") and string.endswith("]"):
# 		return(string[1:-1])
# 	else:
# 		return(string)



# def handle_synonym_in_parentheses(string, min_length):
# 	"""Looks at a string that is suspected to be in a format like "name (othername)". If
# 	that is the case then a list of strings is returned that looks like [name, othername].
# 	This is useful when a column is specifying something like a gene name but a synonym
# 	might be mentioned in the same column in parentheses, so the whole string in that 
# 	column is not useful for searching against as whole. Does not consider text in
# 	parentheses shorter than min_length to be a real synonym, but rather part of the 
# 	name, such as gene_name(t) for example. 

# 	Args:
# 		string (str): Any arbitrary string.
		
# 		min_length (int): Length requirement in characters for synonym to be considered.
	
# 	Returns:
# 		list: A list of strings that are found to be discrete names.
# 	"""
# 	names = []
# 	pattern = r"\(.*?\)"
# 	results = re.findall(pattern, string)
# 	for result in results:
# 		enclosed_string = result[1:-1]
# 		if len(enclosed_string)>=min_length:
# 			string = string.replace(result, "")
# 			names.append(enclosed_string)
# 	names.append(string)
# 	names = [name.strip() for name in names]
# 	return(names)



# def remove_short_tokens(tokens, min_length=2):
# 	"""Remove any string from the list that is doesn't meet threshold for number of characters.
	
# 	Args:
# 	    tokens (list of str): A list of arbitrary strings such as tokens.

# 	    min_length (int, optional): The minimum number of characters for a string to be retained.
	
# 	Returns:
# 	    list of str: The same list but without strings that did not meet the length criteria.
# 	"""
# 	tokens = [token for token in tokens if len(token)>=min_length]
# 	return(tokens)



# def add_prefix(token, prefix):
# 	"""Attaches the passed in prefix argument to the front of the token,
# 	unless the token is an empty string in which case nothing happens
# 	(avoids accidentally making a meaningless token ("") meaningful by
# 	modifying it with an additional component.
# 	Args:
# 		token (str): Any arbitrary string.

# 		prefix (str): Any arbitrary string.

# 	Returns:
# 		str: The token with the prefix added to the beginning of the string.
# 	"""
# 	if len(token) > 0:
# 		return("{}{}".format(prefix, token))
# 	else:
# 		return("")
















# ############### Methods useful for manipulating text descriptions ################



# def append_words(description, words):
# 	"""
# 	Appends all words in a list of words to the end of a description string and returns
# 	the resulting larger string. This is useful in cases where it's desired to generate
# 	variance in word-choice but the structure of the description itself is not important
# 	and can be ignored, such as in using bag-of-words or similar technique.
# 	Args:
# 		description (str): Any string, a description of something.

# 		words (list): Strings to be appended to the description.

# 	Returns:
# 		str: The description with new words appended.
# 	"""
# 	combined_list = [description]
# 	combined_list.extend(words)
# 	description = " ".join(combined_list).strip()
# 	return(description)



# def concatenate_descriptions(*descriptions):
# 	"""
# 	Combines multiple description strings into a single string. Characters which
# 	denote the end of a sentence or fragment are added where necessary so that the
# 	combined description will still be parseable with some other NLP package or 
# 	functions if generating a sentence tree, for example.

# 	Args:
# 		*descriptions: Any number of arbitrary pieces of text.

# 	Returns:
# 		str: The concatenated pieces of text.
# 	"""
# 	descriptions = remove_text_duplicates_retain_order(descriptions)
# 	descriptions = [add_end_char(description.strip()) for description in descriptions]
# 	description = " ".join(descriptions).strip()
# 	description = remove_newlines(description)
# 	return(description)











# ############### Other useful methods for looking at sources of text ################


# def get_ontology_ids(string):
# 	"""
# 	Find all ontology IDs inside of some text. This makes the assumption that
# 	all (and exactly) seven digits of the ontology term ID number are included,
# 	that the abbreviation for the ontology name is in all caps, but makes no 
# 	assumption about the length of the name.

# 	Args:
# 		string (str): Any string of text.
		
# 	Returns:
# 		list: A list of the ontology term IDs mentioned in it.
# 	"""
# 	pattern = r"[A-Z]+:[0-9]{7}"
# 	results = re.findall(pattern, string)
# 	return(results)

































