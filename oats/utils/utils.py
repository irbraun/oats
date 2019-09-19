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











######### Functions that do some logic step that is often needed #########

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
        seq (list): Description
    Returns:
        list: Description
    """
    seen = set()
    seen_add = seen.add
    return([x for x in seq if not (x in seen or seen_add(x))])











######### Random things found to be useful. ########

def to_hms(num_seconds):
	hms_str = time.strftime('%H:%M:%S',time.gmtime(num_seconds))
	return(hms_str)

def function_wrapper(function, args):
	result = function(*args)
	return(result)









######### Reading and writing python objects #########

def save_to_pickle(obj, path):
	pickle.dump(obj, open(path,"wb"))

def load_from_pickle(path):
	obj = pickle.load(open(path,"rb"))
	return(obj)





