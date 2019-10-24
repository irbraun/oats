from itertools import product
from scipy import spatial
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import _pickle as pickle
import time
import json
from collections import defaultdict
from itertools import  chain

from oats.utils import constants









######### Random functions specific to this package #########

def to_abbreviation(name, mapping=None):
	if not mapping == None:
		return(mapping[name])
	else:
		return(constants.ABBREVIATIONS_MAP[name])










######### Functions that do some logic step that is used often #########

def merge_list_dicts(*dicts):
	"""
	Merges dictionaries where the values are lists of items. The behavior is that
	when two mappings are combined, the key is the same but the values are a union
	of the items that were in the two lists.
	
	Args:
		*dicts: Any number of dictionaries to be merged.
	Returns:
		dict: A single dict merged from the input ones.
	"""
	merged_dict = defaultdict(list)
	all_tuples = list(chain.from_iterable([d.items() for d in dicts]))
	for (k,v) in all_tuples:
		merged_dict[k].extend(v)
	for (k,v) in merged_dict.items():
		merged_dict[k] = remove_duplicates_retain_order(v)
	return(merged_dict)


def remove_duplicates_retain_order(seq):
	"""Code credited to https://stackoverflow.com/a/480227.
	
	Args:
		seq (list): Any list of any datatype.
	Returns:
		list: The list in same order but only first occurence of all duplicates retained.
	"""
	seen = set()
	seen_add = seen.add
	return([x for x in seq if not (x in seen or seen_add(x))])



def flatten(l):
	return(list(_recursive_flatten(l)))
def _recursive_flatten(l):
	"""
	https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
	Using itertools.chain.from_iterable() doesn't work for strings because it splits on characters.
	Need to use this function instead when the nested lists contain strings.
	"""
	for x in l:
		if hasattr(x, '__iter__') and not isinstance(x, str):
			for y in flatten(x):
				yield y
		else:
			yield x









######### Random things found to be useful ########

def to_hms(num_seconds):
	hms_str = time.strftime('%H:%M:%S',time.gmtime(num_seconds))
	return(hms_str)

def function_wrapper(function, args):
	result = function(*args)
	return(result)

def function_wrapper_with_duration(function, args):
	""" Call a function and return the result plus duration in seconds.
	Args:
		function (function): Any arbitrary method.
		args (list): The arguments to be sent to the function.
	Returns:
		tuple: The output of the function and runtime in seconds.
	"""
	start_time = time.perf_counter()
	return_value = function(*args)
	total_time = time.perf_counter()-start_time
	return(return_value,total_time)

def print_nested_dict(d, indent=0):
	"""Credited to comment at https://stackoverflow.com/a/3229493.
	Args:
		d (dict): Any nested dictionary.
		indent (int, optional): Number of tabs in the indent.
	"""
	for key,value in d.items():
		print('\t' * indent + str(key))
		if isinstance(value, dict):
			print_nested_dict(value, indent+1)
		else:
			print('\t' * (indent+1) + str(value))








######### Reading and writing python objects #########

def save_to_pickle(obj, path):
	pickle.dump(obj, open(path,"wb"))

def load_from_pickle(path):
	obj = pickle.load(open(path,"rb"))
	return(obj)













