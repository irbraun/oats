from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd



class SquarePairwiseDistances:

	""" An object that contains the results of doing a pairwise comparison within one group of texts.
	
	Attributes:
	    array (TYPE): Description
	    
	    edgelist (TYPE): Description
	    
	    id_to_index (TYPE): Description
	    
	    index_to_id (TYPE): Description
	    
	    metric_str (TYPE): Description
	    
	    vector_dictionary (TYPE): Description
	    
	    vectorizer_object (TYPE): Description
	    
	    vectorizing_function (TYPE): Description
	    
	    vectorizing_function_kwargs (TYPE): Description
	"""
	
	def __init__(self,
		metric_str,
		edgelist,
		vector_dictionary,
		id_to_index,
		index_to_id,
		array,
		vectorizing_function=None,
		vectorizing_function_kwargs=None,
		vectorizer_object=None): 
		"""
		This class is for creating objects to hold the specifications for a pairwise similarity or distance matrax 
		between a set of objects with IDs (such as genes in some dataset), as well as remembering certain information 
		about how the graph was constructed so that a new object of some text or annotations could be placed within 
		the context of this graph without rebuilding the entire graph. To this end, this class also provides a method
		for vectorizing new instances of text in the same way that text was vectorized to build the graph in the
		first place, and a method for finding the k nearest neighbors to a new instance of text that is present in the
		graph. Note that this does not currently include preprocessing, so this class does not remember how the text 
		needs to be preprocessed in order to be fully compatible with these vectors, only how the vectorization is
		done once the text has been preprocessed. 
		
		Args:
		    metric_str (str): A string indicating which distance metric was/should be used, compatible with sklearn.
		    
		    vectorizing_function (function): A function to call to convert text to vector compatible with this graph.
		    
		    vectorizing_function_kwargs (dict of str:obj): A dictionary of keyword arguments that are passed to the vectorizing function.
		    
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format (from,to,value).
		    
		    vector_dictionary (dict of int:int): A mapping between node IDs and vector representation.
		    
		    vectorizer_object (None, optional): The vectorizer object used for embedding each node.
		    
		    id_to_index (dict of int:int): A mapping between node IDs and indices in the distance matrix.
		    
		    index_to_id (dict of int:int): A mapping between indices in the distance matrix and node IDs.
		    
		    array (numpy.array): A numpy array containing all the distance values that were calculated.
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
		self.ids = list(self.id_to_index.keys())
		


		# Make sure that the relationships between the dictionaries and arrays are as expected.
		assert len(self.ids) == len(self.id_to_index)
		assert len(self.ids) == len(self.index_to_id)
		assert len(self.ids) == len(self.vector_dictionary)
		assert len(self.ids) == self.array.shape[0]
		assert len(self.ids) == self.array.shape[1]

		# Making sure the shape of all the required passed in objects are as expected.
		ids_in_edgelist = pd.unique(self.edgelist[["from","to"]].dropna().values.ravel('K'))  	# Get a list of all the IDs mentioned as nodes in the edgelist.
		assert (self.array.shape[0] == self.array.shape[1])										# Check that the produced distance matrix is square.
		assert (self.array.shape[0] == len(ids_in_edgelist))									# Check that there is one ID per row and column of the matrix.
		assert (self.array.shape[0] == len(self.vector_dictionary.keys()))						# Check that there is one ID per key in the vector dictionary.
		assert (self.array.shape[0] == len(self.id_to_index.keys()))							# Check that there is one ID per key in the array index dictionary.

		# Making sure the format of the required passed in objects are as expected.
		assert all([a == b for a, b in zip(self.edgelist.columns, ["from","to","value"])])




	def get_vector(self, text):
		return(self.vectorizing_function(text, **self.vectorizing_function_kwargs))







	# Simple convenience function so you don't have to get the dictionaries from this object to lookup distances.
	def get_distance(self, id1, id2):
		distance = self.array[self.id_to_index[id1], self.id_to_index[id2]]
		return(distance)




	# What if we have a new string of text, and we want to know what the distance of it to everything in this array?
	# Essentially get a new 1 by n slice of the array for just this new text compared to the all the ones that were done here.
	# Note, this does NOT include preprocessing of the string, making sure it's compatible has to be done outside this class.
	def get_distances(self, text):

		# Get the vector representation of this new text.
		new_vector = self.get_vector(text)

		# Produce a 1 by n distance matrix that has one row for the new text, and where the columns match existing matrix.
		dim = len(self.vector_dictionary)
		old_vectors = [self.vector_dictionary[self.index_to_id[idx]] for idx in np.arange(dim)]
		one_by_n_matrix = cdist([new_vector], old_vectors, self.metric_str)

		# Produce and return a mapping between these internal IDs and the distance to the new text.
		id_to_distance = {i:one_by_n_matrix[0,idx] for i,idx in self.id_to_index.items()}
		return(id_to_distance)







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

	""" An object that contains the results of doing a pairwise comparison between two groups of texts.
	
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
		edgelist,
		row_vector_dictionary,
		col_vector_dictionary, 
		id_to_row_index, 
		id_to_col_index,
		row_index_to_id, 
		col_index_to_id,
		array,
		vectorizing_function,
		vectorizing_function_kwargs, 
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
		
		Args:
		    metric_str (str): A string indicating which distance metric was/should be used, compatible with sklearn.
		
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format (from,to,value).
		
		    row_vector_dictionary (dict of int:numpy.array): A mapping between row node IDs and vector representations.
		
		    col_vector_dictionary (dict of int:numpy.array): A mappign between column node IDs and vector representations.
		
		    id_to_row_index (dict of int:int): A mapping between ID integers and row indices in the array.
		
		    id_to_col_index (dict of int:int): A mapping between ID integers and column indices in the array.
		
		    row_index_to_id (dict of int:int): A mapping between row indices in the array and ID integers.
		
		    col_index_to_id (dict of int:int): A mapping between column indices in the array and ID integers.
		
		    array (numpy.array): A numpy array containing all the distance values that were calculated.

		    vectorizing_function (function): A function to call to convert text to vector compatible with this graph.
		
		    vectorizing_function_kwargs (dict of str:obj): A dictionary of keyword arguments that are passed to the vectorizing function.
		
		    vectorizer_object (None, optional): The vectorizer object used for embedding each node.
		
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














		