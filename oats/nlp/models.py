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
from collections import defaultdict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process








def load_word2vec_model(model_path):
	model = gensim.models.word2vec.Word2Vec.load(model_path)
	return(model)



def train_word2vec_model(model_path, training_textfile_path, size=300, sg=1, hs=1, sample=1e-3, window=10, alpha=0.025, workers=5):
	text = gensim.models.word2vec.Text8Corpus(training_textfile_path)
	model = gensim.models.word2vec.Word2Vec(sentences, size=size, sg=sg, hs=hs, sample=sample, window=window, alpha=alpha, workers=workers) 
	model.save(model_path)


