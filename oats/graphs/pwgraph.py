from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd



class SquarePairwiseDistances:

	def __init__(self,
		metric_str,
		vectorizing_function,
		vectorizing_function_kwargs, 
		edgelist,
		vector_dictionary,
		id_to_index,
		index_to_id,
		array,
		vectorizer_object=None): 
		"""
		This class is for creating objects to hold the specifications for a pairwise similarity or distance graph 
		between a set of objects with IDs (such as genes in some dataset), as well as remembering certain information 
		about how the graph was constructed so that a new object of some text or annotations could be placed within 
		the context of this graph without rebuilding the entire graph. To this end, this class also provides a method
		for vectorizing new instances of text in the same way that text was vectorized to build the graph in the
		first place, and a method for finding the k nearest neighbors to a new instance of text that is present in the
		graph. Note that this does not currently include preprocessing, so this class does not remember how the text 
		needs to be preprocessed in order to be fully compatible with these vectors, only how the vectorization is
		done once the text has been preprocessed. 
		
		TODO provide these methods without building the full pairwise distance matrix, if it's not needed.
		
		Args:
		    metric_str (str): A string indicating which distance metric was/should be used, compatible with sklearn.
		    vectorizing_function (function): A function to call to convert text to vector compatible with this graph.
		    vectorizing_function_kwargs (TYPE): Description
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format {from,to,value}.
		    vector_dictionary (dict): A mapping between node IDs and vector representation.
		    vectorizer_object (None, optional): The vectorizer (scipy) object used for embedding each node.
		    id_to_index (None, optional): Description
		    index_to_id (None, optional): Description
		    array (None, optional): A numpy array containing all the distance values that were calculated.
		"""
		self.metric_str = metric_str
		self.vectorizing_function = vectorizing_function
		self.vectorizing_function_kwargs = vectorizing_function_kwargs
		self.edgelist = edgelist
		self.vector_dictionary = vector_dictionary
		self.vectorizer_object = vectorizer_object
		self.id_to_index = id_to_index
		self.index_to_id = index_to_id
		self.array = array 



		# Making sure the shape of all the required passed in objects are as expected.
		ids = pd.unique(self.edgelist[["from","to"]].dropna().values.ravel('K'))  				# Get a list of all the IDs mentioned as nodes in the edgelist.
		assert (self.array.shape[0] == self.array.shape[1])										# Check that the produced distance matrix is square.
		assert (self.array.shape[0] == len(ids))												# Check that there is one ID per row and column of the matrix.
		assert (self.array.shape[0] == len(self.vector_dictionary.keys()))						# Check that there is one ID per key in the vector dictionary.
		assert (self.array.shape[0] == len(self.id_to_index.keys()))							# Check that there is one ID per key in the array index dictionary.

		# Making sure the format of the required passed in objects are as expected.
		assert all([a == b for a, b in zip(self.edgelist.columns, ["from","to","value"])])









	def get_vector(self, text):
		return(self.vectorizing_function(text, **self.vectorizing_function_kwargs))



	def get_nearest_neighbor_ids(self, text, k):
		"""
		Returns a list of k IDs which are the closest to a given string of text. Currently written
		to accept a single string of text, not a list of strings. Also generates the KNN model
		inside this method, TODO move this outside of this method if it it's too slow.
		Args:
		    text (str): Any string of text.
		    k (int): The number of neighbor IDs to return.
		Returns:
		    list: A list of the IDs for the nearest neighbors to the input text.
		"""

		# Creating the KNN model, move this outside this method if too slow.
		knn = NearestNeighbors(n_neighbors=k, metric=self.metric_str)
		ids = list(self.vector_dictionary.keys())
		sample_feature_matrix = np.array([self.vector_dictionary[i] for i in ids])
		knn.fit(sample_feature_matrix)

		# Returning the k nearest neighbors to the vector found for this input text.
		vector = self.get_vector(text)
		neighbor_indices = knn.kneighbors([vector],return_distance=False)[0]
		neighbor_ids = [ids[i] for i in neighbor_indices]
		return(neighbor_ids)













class RectangularPairwiseDistances:

	"""Summary
	
	Attributes:
	    array (TYPE): Description
	    col_index_to_id (TYPE): Description
	    col_vector_dictionary (TYPE): Description
	    edgelist (TYPE): Description
	    id_to_col_index (TYPE): Description
	    id_to_row_index (TYPE): Description
	    metric_str (TYPE): Description
	    row_index_to_id (TYPE): Description
	    row_vector_dictionary (TYPE): Description
	    vectorizer_object (TYPE): Description
	    vectorizing_function (TYPE): Description
	    vectorizing_function_kwargs (TYPE): Description
	"""
	
	def __init__(self,
		metric_str,
		vectorizing_function,
		vectorizing_function_kwargs, 
		edgelist,
		row_vector_dictionary,
		col_vector_dictionary, 
		id_to_row_index, 
		id_to_col_index,
		row_index_to_id, 
		col_index_to_id,
		array,
		vectorizer_object=None):
		"""
		This class is for creating objects to hold the specifications for a pairwise similarity or distance graph 
		between a set of objects with IDs (such as genes in some dataset), as well as remembering certain information 
		about how the graph was constructed so that a new object of some text or annotations could be placed within 
		the context of this graph without rebuilding the entire graph. To this end, this class also provides a method
		for vectorizing new instances of text in the same way that text was vectorized to build the graph in the
		first place, and a method for finding the k nearest neighbors to a new instance of text that is present in the
		graph. Note that this does not currently include preprocessing, so this class does not remember how the text 
		needs to be preprocessed in order to be fully compatible with these vectors, only how the vectorization is
		done once the text has been preprocessed. 
		
		TODO provide these methods without building the full pairwise distance matrix, if it's not needed.
		
		Args:
		    metric_str (str): A string indicating which distance metric was/should be used, compatible with sklearn.
		    vectorizing_function (function): A function to call to convert text to vector compatible with this graph.
		    vectorizing_function_kwargs (TYPE): Description
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format {from,to,value}.
		    row_vector_dictionary (None, optional): A mapping between row node IDs and vector representations.
		    col_vector_dictionary (None, optional): A mappign between column node IDs and vector representations.
		    vectorizer_object (None, optional): The vectorizer (scipy) object used for embedding each node.
		    id_to_row_index (None, optional): A mapping between ID integers and row indices in the array.
		    id_to_col_index (None, optional): A mapping between ID integers and column indices in the array.
		    row_index_to_id (None, optional): A mapping between row indices in the array and ID integers.
		    col_index_to_id (None, optional): A mapping between column indices in the array and ID integers.
		    array (None, optional): A numpy array containing all the distance values that were calculated.
		
		"""
		self.metric_str = metric_str
		self.vectorizing_function = vectorizing_function
		self.vectorizing_function_kwargs = vectorizing_function_kwargs
		self.edgelist = edgelist
		self.row_vector_dictionary = row_vector_dictionary
		self.col_vector_dictionary = col_vector_dictionary 
		self.vectorizer_object = vectorizer_object
		self.id_to_row_index = id_to_row_index
		self.id_to_col_index = id_to_col_index
		self.row_index_to_id = row_index_to_id
		self.col_index_to_id = col_index_to_id
		self.array = array 

		# Making sure the shape of all the required passed in objects are as expected.
		row_ids = pd.unique(self.edgelist[["from"]].dropna().values.ravel('K'))
		col_ids = pd.unique(self.edgelist[["to"]].dropna().values.ravel('K'))
		assert (self.array.shape[0] == len(row_ids))
		assert (self.array.shape[1] == len(col_ids))
		assert (self.array.shape[0] == len(self.row_vector_dictionary.keys()))
		assert (self.array.shape[1] == len(self.col_vector_dictionary.keys()))
		assert (self.array.shape[0] == len(self.id_to_row_index.keys()))
		assert (self.array.shape[1] == len(self.id_to_col_index.keys()))
		assert (self.array.shape[0] == len(self.row_index_to_id.keys()))
		assert (self.array.shape[1] == len(self.col_index_to_id.keys()))

		# Making sure the format of the required passed in objects are as expected.
		assert all([a == b for a, b in zip(self.edgelist.columns, ["from","to","value"])])













	def get_vector(self, text):
		return(self.vectorizing_function(text, **self.vectorizing_function_kwargs))



	def get_nearest_neighbor_ids(self, text, k):
		"""
		Returns a list of k IDs which are the closest to a given string of text. Currently written
		to accept a single string of text, not a list of strings. Also generates the KNN model
		inside this method, TODO move this outside of this method if it it's too slow.
		Args:
		    text (str): Any string of text.
		    k (int): The number of neighbor IDs to return.
		Returns:
		    list: A list of the IDs for the nearest neighbors to the input text.
		"""

		# Creating the KNN model, move this outside this method if too slow.
		knn = NearestNeighbors(n_neighbors=k, metric=self.metric_str)
		ids = list(self.vector_dictionary.keys())
		sample_feature_matrix = np.array([self.vector_dictionary[i] for i in ids])
		knn.fit(sample_feature_matrix)

		# Returning the k nearest neighbors to the vector found for this input text.
		vector = self.get_vector(text)
		neighbor_indices = knn.kneighbors([vector],return_distance=False)[0]
		neighbor_ids = [ids[i] for i in neighbor_indices]
		return(neighbor_ids)














		