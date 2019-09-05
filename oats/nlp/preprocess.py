from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import re
from collections import defaultdict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process





############### General-purpose functions for preprocessing text ################



def get_clean_description(description):
	description = remove_punctuation(description)
	description = description.lower()
	return(description)


def get_clean_token_list(description):
	description = remove_punctuation(description)
	token_list = description.lower().split()
	return(token_list)


def remove_punctuation(text):
	translator = str.maketrans('', '', string.punctuation)
	return(text.translate(translator))


def remove_newlines(text):
	text = text.replace("\n", " ")
	text = text.replace("\t", " ")
	text = text.replace("\r", " ")
	return(text)


def add_end_tokens(description):
	if len(description) > 0:
		last_character = description[len(description)-1]
		end_tokens = [".", ";"]
		if not last_character in end_tokens:
			description = description+"."
	return(description)










############### Things useful for converting between strings and lists ################


def concatenate_with_bar_delim(*tokens):
	"""
	Concatenates any number of passed in tokens with a bar character and returns the 
	resulting string. This is useful for preparing things like gene names for entry
	into a table that could be written to a csv where multiple elements need to go 
	into a single column but a standard delimiter like a comma should not be used.
	Some of the tokens passed in can strings that are already delimited with a bar 
	character. These are treated as seperate elements that should be incorporated 
	into the string that is returned, trailing and leading bars are handled.
	Args:
	    *tokens: Description
	Returns:
	    TYPE: Description
	"""
	tokens = [token.split("|") for token in tokens]
	tokens = itertools.chain.from_iterable(tokens)
	tokens = filter(None, tokens)
	tokens = [token.strip() for token in tokens]
	tokens = list(set(tokens)) # Remove duplicates that may have been introduced.
	joined = "|".join(tokens).strip()
	joined = remove_newlines(joined)
	return(joined)



def other_delim_to_bar_delim(string, delim):
	"""Convert string delimited with some character (semicolon, comma, etc) to one delimited with bars.
	"""
	if not len(delim) == 1:
		raise ValueError("delimiter {} is not a single character".format(delim))
	tokens = string.split(delim)
	tokens = [token.strip() for token in tokens]
	joined = "|".join(tokens)
	joined = remove_newlines(joined)
	return(joined)












############### Functions useful for working with gene names ################



def remove_enclosing_brackets(string):
	"""If brackets enclose the string, remove them.
	"""
	if string.startswith("[") and string.endswith("]"):
		return(string[1:-1])
	else:
		return(string)



def remove_character(string, char):
	"""Remove all occurences of a particular character from a string.
	"""
	string = string.replace(char,"")
	return(string)



def handle_synonym_in_parentheses(string, min_length):
	""" 
	Looks at a string that is suspected to be in a format like "name (othername)". If
	that is the case then a list of strings is returned that looks like [name, othername].
	This is useful when a column is specifying something like a gene name but a synonym
	might be mentioned in the same column in parentheses, so the whole string in that 
	column is not useful for searching against as whole. Does not consider text in
	parentheses shorter than min_length to be a real synonym, but rather part of the 
	name, something like genename(t) for example. 
	Args:
	    string (str): Any string.
	    min_length (int): Length requirement in characters for synonym to be considered.
	Returns:
	    list: All strings that are supposed to be discrete components.
	"""
	names = []
	pattern = r"\(.*?\)"
	results = re.findall(pattern, string)
	for result in results:
		enclosed_string = result[1:-1]
		if len(enclosed_string)>=min_length:
			string = string.replace(result, "")
			names.append(enclosed_string)
	names.append(string)
	names = [name.strip() for name in names]
	return(names)



def remove_short_tokens(tokens, min_length=2):
	"""Remove any string from the list that is doesn't meet threshold for number of characters.
	"""
	tokens = [token for token in tokens if len(token)>=min_length]
	return(tokens)




def add_prefix(token, prefix):
	"""
	Attaches the passed in prefix argument to the front of the token,
	unless the token is an empty string in which case nothing happens
	(avoids accidentally making a meaningless token ("") meaningful by
	modifying it with an additional component.
	Args:
	    token (str): Any string.
	    prefix (str): Any string.
	Returns:
	    str: The token with the prefix added to the beginning.
	"""
	if len(token) > 0:
		return("{}{}".format(prefix, token))
	else:
		return("")













############### Methods useful for manipulating text descriptions ################




def append_words(description, words):
	"""
	Appends all words in a list of words to the end of a description string and returns
	the resulting larger string. This is useful in cases where it's desired to generate
	variance in word-choice but the structure of the description itself is not important
	and can be ignored, such as in using bag-of-words or similar technique.
	Args:
	    description (str): Any string, a description of something.
	    words (list): Strings to be appended to the description.
	Returns:
	    str: The description with new words appended.
	"""
	combined_list = [description]
	combined_list.extend(words)
	description = " ".join(combined_list).strip()
	return(description)



def concatenate_descriptions(*descriptions):
	"""
	Combines multiple description strings into a single string. Characters which
	denote the end of a sentence or fragment are added where necessary so that the
	combined description will still be parseable with some other NLP package or 
	functions if generating a sentence tree or something like that.
	Args:
	    *descriptions: Description
	Returns:
	    TYPE: Description
	"""
	descriptions = [add_end_tokens(description) for description in descriptions]
	description = " ".join(descriptions).strip()
	description = remove_newlines(description)
	return(description)











############### Other useful methods for looking at source data ################


def get_ontology_ids(string):
	"""
	Find all ontology IDs inside of some text. This makes the assumption that
	all (and exactly) seven digits of the ontology term ID number are included,
	that the abbreviation for the ontology name is in all caps, but makes no 
	assumption about the length of the name.
	Args:
	    string (str): Any string of text.
	Returns:
	    list: A list of the ontology term IDs mentioned in it.
	"""
	pattern = r"[A-Z]+:[0-9]{7}"
	results = re.findall(pattern, string)
	return(results)


















