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
import networkx as nx
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier






class Graph:



	def __init__(self, df, value):
		self.ids_in_graph = self._get_ids_in_graph(df)
		self.id_to_array_index = self._get_mapping_from_id_to_index()
		self.np_array = self._to_np_array(df, value)





	def get_value(self, id_1, id_2):
		""" Get the value for the edge between these two nodes in the graph.
		Args:
		    id_1 (TYPE): Description
		    id_2 (TYPE): Description
		Returns:
		    TYPE: Description
		"""
		return(self.np_array[self.id_to_array_index[id_1]][self.id_to_array_index[id_2]])



	def get_ids_in_graph(self):
		return(self.ids_in_graph)






	def _get_ids_in_graph(self, df):
		"""
		Get the IDs that are specified in the dataframe for this graph. These are used as the
		complete list of node names which to retain references to when building the matrix. 
		Args:
		    df (TYPE): Description
		Returns:
		    TYPE: Description
		"""
		ids_set = set()
		ids_set.update(list(pd.unique(df["from"].values)))
		ids_set.update(list(pd.unique(df["to"].values)))
		ids = list(ids_set)
		return(ids)





	def _get_mapping_from_id_to_index(self):
		"""
		Produce a mapping between each node name (ID) and an integer which corresponds to an
		index along the axis of the matrix. This allows for looking up specific values in the 
		matrix by providing IDs and then translating those to indices.
		
		Returns:
		    TYPE: Description
		"""
		ids = self.ids_in_graph
		mapping = {}
		for idx in range(0,len(ids)):
			mapping[ids[idx]] = idx
		return(mapping)





	def _to_np_array(self, df, value):
		"""	
		This function assumes that the graph described is undirected, otherwise this will be 
		setting and resetting the same values multiple times. Does not assume that every 
		edge in the graph is specified (graph does not need to be complete in the data frame
		in order to get back a full matrix here, the default for missing value is 0).
		
		Args:
		    df (TYPE): Description
		    value (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		ids = self.ids_in_graph
		dim = len(ids)
		arr = np.zeros(shape=(dim,dim))

		# What are the positions of the important information columns in the dataframes?
		# The indices that are returned from the column list are one less than the actual
		# indices we need from the row tuples, because row tuples include the index value
		# at position 0 and everything else moved up one spot.
		from_pos = df.columns.get_loc("from")+1
		to_pos = df.columns.get_loc("to")+1
		value_pos = df.columns.get_loc(value)+1

		for row in df.itertuples():
			arr[self.id_to_array_index[row[from_pos]]][self.id_to_array_index[row[to_pos]]] = row[value_pos]
			arr[self.id_to_array_index[row[to_pos]]][self.id_to_array_index[row[from_pos]]] = row[value_pos]
		return(arr)

		#This nly works the ordering of the nodes in the edge specification is fixed.
		# Because not using verify_integrity, there also could  be problems if there is an 
		# issue with the data frame that is passed in, if the edges that are specified are 
		# not unique.
		"""
		df.set_index(keys=["from","to"], drop=True, inplace=True, verify_integrity=False)
		for id_1,id_2 in list(itertools.combinations_with_replacement(ids,2)):
			index_key = (id_1, id_2)
			v = df.loc[index_key][value]
			arr[self.id_to_array_index[id_1]][self.id_to_array_index[id_2]] = v
			arr[self.id_to_array_index[id_2]][self.id_to_array_index[id_1]] = v
		return(arr)
		"""








