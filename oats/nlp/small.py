import itertools
import re
import string
from collections import defaultdict
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces

from oats.utils.utils import remove_duplicates_retain_order







def remove_punctuation(text):
	"""Remove all punctuation from a piece of text.
	
	Args:
	    text (str): Any piece of text.
	
	Returns:
	    str: That text without any characters that were punctuation.
	"""
	translator = str.maketrans('', '', string.punctuation)
	return(text.translate(translator))







def remove_newlines(text):
	"""Remove all newline characters from a piece of text.
	
	Args:
	    text (str): Any piece of text.
	
	Returns:
	    str: That text without newline characters.
	"""
	text = text.replace("\n", " ")
	text = text.replace("\t", " ")
	text = text.replace("\r", " ")
	return(text)







def add_end_character(text):
	"""Adds a period to the end of the text. This could be useful when concatentating
	text while still retaining the sentence or phrase boundaries taken into account by
	other processing steps such as part-of-speech analysis. Accounts for text that 
	already ends in periods or semicolons or is an empty string.
	
	Args:
	    text (str): Any piece of text.
	
	Returns:
	    str: That text with a period added to the end.
	"""
	if len(text) > 0:
		last_character = text[len(text)-1]
		end_tokens = [".", ";"]
		if not last_character in end_tokens:
			text = text+"."
	return(text)








def capitalize_sentence(text):
	"""Makes the first character of a text string captial if it is a letter.
	
	Args:
	    text (str): Any arbitrary text string.
	
	Returns:
	    str: The text string with the first letter capitalized.
	"""
	if len(text)>0:
		modified_text = "{}{}".format(text[0].upper(), text[1:])
		return(modified_text)
	else:
		return(text)








def add_prefix_safely(token, prefix):
	"""Attaches the passed in prefix argument to the front of the token, unless the token 
	is an empty string in which case nothing happens and the token is returned unchaged.
	This can be important for avoiding accidentally making a meaningless token meaningful
	by modifying it with an additional text component.

	Args:
		token (str): Any arbitrary string.

		prefix (str): Any arbitrary string.

	Returns:
		str: The token with the prefix added to the beginning of the string.
	"""
	if len(token) > 0:
		return("{}{}".format(prefix, token))
	else:
		return("")









def remove_enclosing_brackets(text):
	"""Removes square brackets if they are enclosing the text string.
	
	Args:
	    text (str): Any arbitrary string.
	
	Returns:
	    str: The same string but with the enclosing brackets removed if they were there.
	"""
	if text.startswith("[") and text.endswith("]"):
		return(text[1:-1])
	else:
		return(text)











def get_ontology_ids(text):
	"""
	Find all ontology IDs inside of some text. This makes the assumption that
	all (and exactly) seven digits of the ontology term ID number are included,
	that the abbreviation for the ontology name is in all caps, but makes no 
	assumption about the length of the name.

	Args:
		text (str): Any string of text.
		
	Returns:
		list: A list of the ontology term IDs mentioned in it.
	"""
	pattern = r"[A-Z]+:[0-9]{7}"
	results = re.findall(pattern, text)
	return(results)

































