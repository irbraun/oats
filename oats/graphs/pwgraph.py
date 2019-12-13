



class PairwiseGraph:

	def __init__(self, edgelist, vector_dictionary=None, vectorizer_object=None, id_to_index=None, index_to_id=None, array=None):
		"""
		This class is for creating objects to hold the specifications for a pairwise
		similarity or distance graph between a set of objects with IDs (such as genes
		in some dataset), as well as remembering certain information about how the 
		graph was constructed so that a new object of some text or annotations could
		be placed within the context of this graph without rebuilding the entire graph.
		
		Args:
		    edgelist (pandas.DataFrame): Each row is an edge in the graph with format {from,to,value}.
		    vector_dictionary (None, optional): A mapping between node IDs and vector representation.
		    vectorizer_object (None, optional): The vectorizer (scipy) object used for embedding each node.
		    id_to_index (None, optional): A mapping between ID integers and indices in the array.
		    index_to_id (None, optional): A mapping between indices in the array and ID integers.
		    array (None, optional): A numpy array where the 
		
		"""
		self.edgelist = edgelist
		self.vector_dictionary = vector_dictionary
		self.vectorizer_object = vectorizer_object
		self.id_to_index = id_to_index
		self.index_to_id = index_to_id
		self.array = array
