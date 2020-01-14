



class PairwiseGraph:

	def __init__(self, 
		edgelist, 
		vector_dictionary=None,
		row_vector_dictionary=None,
		col_vector_dictionary=None, 
		vectorizer_object=None, 
		id_to_row_index=None, 
		id_to_col_index=None,
		row_index_to_id=None, 
		col_index_to_id=None,
		array=None):
		"""
		This class is for creating objects to hold the specifications for a pairwise
		similarity or distance graph between a set of objects with IDs (such as genes
		in some dataset), as well as remembering certain information about how the 
		graph was constructed so that a new object of some text or annotations could
		be placed within the context of this graph without rebuilding the entire graph.
		
		Args:
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format {from,to,value}.
		    vector_dictionary (None, optional): A mapping between node IDs and vector representation.
		    row_vector_dictionary (None, optional): A mapping between row node IDs and vector representations.
		    col_vector_dictionary (None, optional): A mappign between column node IDs and vector representations.
		    vectorizer_object (None, optional): The vectorizer (scipy) object used for embedding each node.
		    id_to_row_index (None, optional): A mapping between ID integers and row indices in the array.
		    id_to_col_index (None, optional): A mapping between ID integers and column indices in the array.
		    row_index_to_id (None, optional): A mapping between row indices in the array and ID integers.
		    col_index_to_id (None, optional): A mapping between column indices in the array and ID integers.
		    array (None, optional): A numpy array containing all the distance values that were calculated.
		
		"""
		self.edgelist = edgelist
		self.vector_dictionary = vector_dictionary
		self.row_vector_dictionary = row_vector_dictionary,
		self.col_vector_dictionary = col_vector_dictionary, 
		self.vectorizer_object = vectorizer_object
		self.id_to_row_index = id_to_row_index
		self.id_to_col_index = id_to_col_index
		self.row_index_to_id = row_index_to_id
		self.col_index_to_id = col_index_to_id
		self.array = array

		