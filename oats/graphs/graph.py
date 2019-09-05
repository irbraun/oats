from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from functools import reduce
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier





class Graph:



	def __init__(self, df, value):
		self.ids_in_graph = self._get_ids_in_graph(df)
		self.id_to_array_index = self._get_mapping_from_id_to_index(df)
		self.np_array = self._to_np_array(df, value)


	def _get_ids_in_graph(self, df):
		ids_set = set()
		ids_set.update(list(pd.unique(df["from"].values)))
		ids_set.update(list(pd.unique(df["to"].values)))
		ids = list(ids_set)
		return(ids)


	def _get_mapping_from_id_to_index(self, df):
		ids = self.ids_in_graph
		mapping = {}
		for idx in range(0,len(ids)):
			mapping[ids[idx]] = idx
		return(mapping)




	# Makes the assumption that if an edge is mentioned twice (as if it was directional), the value will be the same.
	# TODO change it so that this is explicity checked for and raises errors where appropriate.
	def _to_np_array(self, df, value):
		ids = self.ids_in_graph
		dim = len(ids)
		arr = np.zeros(shape=(dim,dim))
		for (id_1,id_2) in itertools.product(ids,ids):
			relevant_rows = df[((df["from"]==id_1) & (df["to"]==id_2)) | ((df["from"]==id_2) & (df["to"]==id_1))]
			v = relevant_rows[value].values[0]

			# What if this ID pair is used twice, as if the edge was directed?

			# What if this ID pair was not mentioned at all? Should never occur.

			# Populate the corresponding cells of the array.
			arr[self.id_to_array_index[id_1]][self.id_to_array_index[id_2]] = v
			arr[self.id_to_array_index[id_2]][self.id_to_array_index[id_1]] = v

		return(arr)



	def get_value(self, id_1, id_2):
		return(self.np_array[self.id_to_array_index[id_1]][self.id_to_array_index[id_2]])
