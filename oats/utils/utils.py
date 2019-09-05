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

from oats.utils import constants



def function_wrapper(function, args):
	result = function(*args)
	return(result)





def to_hms(num_seconds):
	hms_str = time.strftime('%H:%M:%S',time.gmtime(num_seconds))
	return(hms_str)




def to_abbreviation(name, mapping=None):
	if not mapping == None:
		return(mapping[name])
	else:
		return(constants.ABBREVIATIONS_MAP[name])








def save_to_pickle(obj, path):
	pickle.dump(obj, open(path,"wb"))

def load_from_pickle(path):
	obj = pickle.load(open(path,"rb"))
	return(obj)





