from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA

from oats.utils.utils import flatten
from oats.graphs.pwgraph import SquarePairwiseDistances
from oats.graphs.pwgraph import RectangularPairwiseDistances

from oats.graphs._utils import _square_adjacency_matrix_to_edgelist
from oats.graphs._utils import _rectangular_adjacency_matrix_to_edgelist
from oats.graphs._utils import _strings_to_count_vectors				
from oats.graphs._utils import _strings_to_tfidf_vectors
from oats.graphs._utils import _infer_document_vector_from_bert			# Is actually called when finding distance matrices.
from oats.graphs._utils import _infer_document_vector_from_word2vec		# Is actually called when finding distance matrices.
from oats.graphs._utils import _infer_document_vector_from_doc2vec		# Is actually called when finding distance matrices.
from oats.graphs._utils import _get_ngrams_vector						# Not called, just a wrapper function to remember a vectorization scheme.
from oats.graphs._utils import _get_annotations_vector					# Not called, just a wrapper function to remember a vectorization scheme.
from oats.graphs._utils import _get_topic_model_vector 					# Not called, just a wrapper function to remember a vectorization scheme.







# Description of the distance functions provided here. 

# Category 1: pairwise_square_[method](...)
# These functions take a single dictionary mapping IDs to text or similar objects and finds 
# the pairwise distances between all of elements in that dictionary using whatever the method
# is that is specific to that function. This produces a square matrix of distances that is 
# symmetrical along the diagonal.

# Category 2: pairwise_rectangular_[method](...)
# These functions take two different dicitonaries mappings IDs to text or similar objects and 
# finds the pairwise distances between all combinations of an element from one group and an
# element from the other group, making the calculation in a way specific whatever the method 
# or approach for that function is. This produces a rectangular matrix of distances. The rows
# of that matrix correspond to the elements form the first dictionary, and the columns to the
# elements from the second dictionary. In edgelist form, the "from" column refers to IDs from
# the first dictionary, and the "to" column refers to IDs from the second dictionary. 

# Categery 3: elemwise_list_[method](...)
# These functions take two lists of text or similar objects that are of the exact same length
# and returns a list of distance values calculated based on the method or approach for that 
# particular function. The distance value in position i of the returned list is the distance 
# found between the element at position i in the first list and the element at position i in 
# the second list.














def strings_to_numerical_vectors(*strs, tfidf=False, **kwargs):
	"""Create a vector embedding for each passed in text string.
	
	Args:
	    strs (list of str): All of the string arguments to find numerical embeddings for.
		
		lowercase (bool, optional). A keyword arg for sklearn.feature_extraction.text.TfidfVectorizer and sklearn.feature_extraction.text.CountVectorizer.
		True by default. All the text is normalized to lowercase.

	    tfidf (bool, optional): Description
	    **kwargs: Description
	
	Returns:
	    TYPE: Description
	"""
	if tfidf:
		return(_strings_to_tfidf_vectors(*strs, **kwargs))
	else:
		return(_strings_to_count_vectors(*strs, **kwargs))












def pairwise_square_precomputed_vectors(ids_to_vectors, metric):
	"""docstring
	"""

	# Remember vectors for each ID and their position in the distance matrix.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,vector in ids_to_vectors.items():
		index_in_matrix = len(vectors)
		vectors.append(vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix
	

	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = ids_to_vectors
	

	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = None,	
		vectorizing_function_kwargs = None,			
		edgelist = edgelist,						
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = None,
		id_to_index = id_to_index_in_matrix,
		index_to_id=index_in_matrix_to_id, 
		array=matrix))






def pairwise_square_doc2vec(model, ids_to_texts, metric):
	"""
	Find distance between strings of text in some input data using Doc2Vec. Note that only 
	very simple preprocessing is done here (case normalizing and splitting on whitespace)
	so any preprocessing steps on the text strings should be done prior to passing them in
	a dictionary to this function.
	
	Args:
	    model (gensim.models.doc2vec): An already loaded Doc2Vec model from a file or training.
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine). 
	
	Returns:
	   	oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.
	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		inferred_vector = _infer_document_vector_from_doc2vec(description, model)
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix
	
	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	

	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_doc2vec,		
		vectorizing_function_kwargs = {"model":model},			
		edgelist = edgelist,						
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = None,
		id_to_index = id_to_index_in_matrix,
		index_to_id = index_in_matrix_to_id, 
		array=matrix))





def pairwise_square_word2vec(model, ids_to_texts, metric, method="mean"):
	"""	
	Find distance between strings of text in some input data using Word2Vec. Note that only 
	very simple preprocessing is done here (case normalizing and splitting on whitespace)
	so any preprocessing steps on the text strings should be done prior to passing them in
	a dictionary to this function. Note that if no words in a description are in the model 
	vocabulary, then a random word will be selected to represent the text. This avoids using 
	one default value which will force all these descriptions to cluster, and prevents an 
	error being raised due to no vector appearing. This should rarely or never happen as 
	long as the text has been preprocessed into reasonable tokens and the model is large.
	
	Args:
	    model (gensim.models.word2vec): An already loaded Word2Vec model from a file or training.
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine). 
	    method (str, optional): Should the word embeddings be combined with mean or max.
	
	Returns:
	   	oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.
	
	Raises:
	    Error: The 'method' argument has to be one of "mean" or "max".
	"""
	
	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		vector = _infer_document_vector_from_word2vec(description, model, method)
		index_in_matrix = len(vectors)		
		vectors.append(vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix


	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	

	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_word2vec,
		vectorizing_function_kwargs = {"model":model, "method":method},
		edgelist = edgelist,
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = None,
		id_to_index=id_to_index_in_matrix,
		index_to_id=index_in_matrix_to_id, 
		array=matrix))







def pairwise_square_bert(model, tokenizer, ids_to_texts, metric, method, layers):
	"""
	Find distance between strings of text in some input data using Doc2Vec. The preprocessing
	done to the text strings here is complex, and uses the passed in tokenizer object as well.
	For this reason, in most cases the text passed in to this method should be the raw
	relatively unprocessed sentence of interest. Splitting up of multiple sentences is handled
	in the helper function for this function.
	
	Args:
	    model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
	    tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    method (str): A string indicating how layers for a token should be combined (concat or sum).
	    layers (int): An integer saying how many layers should be used for each token.
	
	Returns:
	   	oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.

	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		inferred_vector = _infer_document_vector_from_bert(description, model, tokenizer, method, layers)
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}


	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_bert,
		vectorizing_function_kwargs = {"model":model, "tokenizer":tokenizer, "method":method, "layers":layers},
		edgelist = edgelist,
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = None,
		id_to_index=id_to_index_in_matrix, 
		index_to_id=index_in_matrix_to_id, 
		array=matrix))








def pairwise_square_ngrams(ids_to_texts, metric, tfidf=False, **kwargs):
	"""
	Find distance between strings of text in some input data using n-grams. Note that only 
	very simple preprocessing is done after this point (splitting on whitespace only) so 
	all processing of the text necessary should be done prio to passing to this function.
	
	Args:
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    tfidf (bool, optional): Whether to use TFIDF weighting or not.
	    **kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer().
	
	Returns:
	    oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.
	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	descriptions = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix


	# Apply distance metric over all the vectors to yield a matrix.
	vectors,vectorizer = strings_to_numerical_vectors(*descriptions, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}
	

	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_ngrams_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer},
		edgelist = edgelist,
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_index=id_to_index_in_matrix,
		index_to_id=index_in_matrix_to_id, 
		array=matrix))







def pairwise_square_topic_model(ids_to_texts, metric, seed=124134, num_topics=10, algorithm="lda", **kwargs):
	"""
	docstring
	
	Args:
	    ids_to_texts (TYPE): Description
	    metric (TYPE): Description
	    seed (int, optional): Description
	    num_topics (TYPE): Description
	    algorithm (str, optional): Description
	    **kwargs: Description
	
	Returns:
	    TYPE: Description
	
	Raises:
	    ValueError: Description
	
	"""

	# Fitting the topic model using the provided parameters and this dataset of text descriptions.
	vectorizer = TfidfVectorizer(**kwargs)
	if algorithm.lower() == "lda":
		model = LDA(n_components=num_topics, random_state=seed)
	elif algorithm.lower() == "nmf":
		model = NMF(n_components=num_topics, random_state=seed)
	else:
		raise ValueError("algorithm argument is invalid")
	
	# Infer vectors for each string of text and remember mapping to the IDs.
	descriptions = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,description in ids_to_texts.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	ngram_vectors = vectorizer.fit_transform(descriptions).toarray()
	topic_vectors = model.fit_transform(ngram_vectors)
	matrix = squareform(pdist(topic_vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(topic_vectors)}
	
	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_topic_model_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "topic_model":model},
		edgelist = edgelist,
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_index=id_to_index_in_matrix,
		index_to_id=index_in_matrix_to_id, 
		array=matrix))







def pairwise_square_annotations(ids_to_annotations, ontology, metric, tfidf=False, **kwargs):
	"""
	Find distance between nodes of interest in the input dictionary based on the overlap in the
	ontology terms that are mapped to those nodes. The input terms for each ID are in the format
	of lists of term IDs. All inherited terms by all these terms will be added in this function 
	using the provided ontology object so that each node will be represented by the union of all 
	the terms inherited by the terms annotated to it. After that step, the term IDs are simply 
	treated as words in a vocabulary, and the same approach as with n-grams is used to generate 
	the distance matrix.
	
	Args:
	    ids_to_annotations (dict): A mapping between IDs and a list of ontology term ID strings.
	    ontology (Ontology): Ontology object with all necessary fields.
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    tfidf (bool, optional): Whether to use TFIDF weighting or not.
	    **kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer()
	
	Returns:
	    oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.

	"""

	# Infer vectors for each string of text and remember mapping to the IDs.
	joined_term_strings = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,term_list in ids_to_annotations.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		index_in_matrix = len(joined_term_strings)
		joined_term_strings.append(joined_term_string)
		index_in_matrix_to_id[index_in_matrix] = identifier
		id_to_index_in_matrix[identifier] = index_in_matrix

	# Apply distance metric over all the vectors to yield a matrix.
	vectors,vectorizer = strings_to_numerical_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	matrix = squareform(pdist(vectors,metric))
	edgelist = _square_adjacency_matrix_to_edgelist(matrix, index_in_matrix_to_id)
	id_to_vector_dict = {index_in_matrix_to_id[i]:vector for i,vector in enumerate(vectors)}


	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(SquarePairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_annotations_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "ontology":ontology},
		edgelist = edgelist,
		vector_dictionary = id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_index=id_to_index_in_matrix, 
		index_to_id=index_in_matrix_to_id, 
		array=matrix))





































def pairwise_rectangular_precomputed_vectors(ids_to_vectors_1, ids_to_vectors_2, metric):
	"""docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,vector in ids_to_vectors_1.items():
		vectors.append(vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,vector in ids_to_vectors_2.items():
		vectors.append(vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_vectors_1)]
	col_vectors = all_vectors[len(ids_to_vectors_1):]
	row_id_to_vector_dict = ids_to_vectors_1
	col_id_to_vector_dict = ids_to_vectors_2
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)



	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = None,		
		vectorizing_function_kwargs = None,		
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = None,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))










def pairwise_rectangular_doc2vec(model, ids_to_texts_1, ids_to_texts_2, metric):
	"""docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		inferred_vector = _infer_document_vector_from_doc2vec(description, model)
		vectors.append(inferred_vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		inferred_vector = _infer_document_vector_from_doc2vec(description, model)
		vectors.append(inferred_vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)



	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_doc2vec,		
		vectorizing_function_kwargs = {"model":model},		
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = None,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))








def pairwise_rectangular_word2vec(model, ids_to_texts_1, ids_to_texts_2, metric, method="mean"):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		vector = _infer_document_vector_from_word2vec(description, model, method)
		vectors.append(vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		vector = _infer_document_vector_from_word2vec(description, model, method)
		vectors.append(vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	

	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_word2vec,
		vectorizing_function_kwargs = {"model":model, "method":method},
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = None,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))










def pairwise_rectangular_bert(model, tokenizer, ids_to_texts_1, ids_to_texts_2, metric, method, layers):
	"""
	docstring
	"""
	vectors = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0	
	for identifier,description in ids_to_texts_1.items():
		inferred_vector = _infer_document_vector_from_bert(description, model, tokenizer, method, layers)
		vectors.append(inferred_vector)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		inferred_vector = _infer_document_vector_from_bert(description, model, tokenizer, method, layers)
		vectors.append(inferred_vector)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors = vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]
	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)
	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	

	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _infer_document_vector_from_bert,
		vectorizing_function_kwargs = {"model":model, "tokenizer":tokenizer, "method":method, "layers":layers},
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = None,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))





def pairwise_rectangular_ngrams(ids_to_texts_1, ids_to_texts_2, metric, tfidf=False, **kwargs):
	"""
	docstring
	"""
	descriptions = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0
	for identifier,description in ids_to_texts_1.items():
		descriptions.append(description)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		descriptions.append(description)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	all_vectors,vectorizer = strings_to_numerical_vectors(*descriptions, tfidf=tfidf, **kwargs)
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]

	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)

	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	
	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_ngrams_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer},
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))










def pairwise_rectangular_topic_model(ids_to_texts_1, ids_to_texts_2, metric, seed=124134, num_topics=10, algorithm="LDA", **kwargs):
	"""
	docstring
	"""


	# Fitting the topic model using the provided parameters and this dataset of text descriptions.
	vectorizer = TfidfVectorizer(**kwargs)
	if algorithm.lower() == "lda":
		model = LDA(n_components=num_topics, random_state=seed)
	elif algorithm.lower() == "nmf":
		model = NMF(n_components=num_topics, random_state=seed)
	else:
		raise ValueError("algorithm argument is invalid")


	descriptions = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0
	for identifier,description in ids_to_texts_1.items():
		descriptions.append(description)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,description in ids_to_texts_2.items():
		descriptions.append(description)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	# Apply distance metric over all the vectors to yield a matrix.
	ngram_vectors = vectorizer.fit_transform(descriptions).toarray()
	topic_vectors = model.fit_transform(ngram_vectors)
	all_vectors = topic_vectors
	row_vectors = all_vectors[:len(ids_to_texts_1)]
	col_vectors = all_vectors[len(ids_to_texts_1):]

	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)

	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	
	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_topic_model_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "topic_model":model},
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))





















def pairwise_rectangular_annotations(ids_to_annotations_1, ids_to_annotations_2, ontology, metric, tfidf=False, **kwargs):
	"""
	docstring
	"""
	joined_term_strings = []
	row_index_in_matrix_to_id = {}
	col_index_in_matrix_to_id = {}
	id_to_row_index_in_matrix = {}
	id_to_col_index_in_matrix = {}

	row_in_matrix = 0
	for identifier,term_list in ids_to_annotations_1.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		row_index_in_matrix_to_id[row_in_matrix] = identifier
		id_to_row_index_in_matrix[identifier] = row_in_matrix 
		row_in_matrix = row_in_matrix+1

	col_in_matrix = 0
	for identifier,term_list in ids_to_annotations_2.items():
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)
		col_index_in_matrix_to_id[col_in_matrix] = identifier
		id_to_col_index_in_matrix[identifier] = col_in_matrix 
		col_in_matrix = col_in_matrix+1

	# Find all the pairwise values for the distance matrix.
	all_vectors,vectorizer = strings_to_numerical_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	row_vectors = all_vectors[:len(ids_to_annotations_1)]
	col_vectors = all_vectors[len(ids_to_annotations_1):]

	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
	matrix = cdist(row_vectors, col_vectors, metric)	
	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)


	return(RectangularPairwiseDistances(
		metric_str = metric,
		vectorizing_function = _get_annotations_vector,
		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "ontology":ontology},
		edgelist = edgelist,
		row_vector_dictionary = row_id_to_vector_dict,
		col_vector_dictionary = col_id_to_vector_dict,
		vectorizer_object = vectorizer,
		id_to_row_index=id_to_row_index_in_matrix, 
		id_to_col_index=id_to_col_index_in_matrix,
		row_index_to_id=row_index_in_matrix_to_id, 
		col_index_to_id=col_index_in_matrix_to_id,
		array=matrix))























def elemwise_list_precomputed_vectors(vector_list_1, vector_list_2, metric_function):
	"""
	docstring
	"""
	assert len(vector_list_1) == len(vector_list_2)
	vector_pairs = zip(vector_list_1, vector_list_2)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(vector_list_1)
	return(distances_list)







def elemwise_list_doc2vec(model, text_list_1, text_list_2, metric_function):
	"""
	docstring
	"""
	assert len(text_list_1) == len(text_list_2)
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = [_infer_document_vector_from_doc2vec(description, model) for description in descriptions]
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	assert len(list_1_vectors) == len(list_2_vectors)
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(list_1_vectors)
	return(distances_list)




def elemwise_list_word2vec(model, text_list_1, text_list_2, metric_function, method="mean"):
	"""
	docstring
	"""
	assert len(text_list_1) == len(text_list_2)
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = []
	all_vectors = [_infer_document_vector_from_word2vec(description, model, method) for description in descriptions]
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	assert len(list_1_vectors) == len(list_2_vectors)
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(list_1_vectors)
	return(distances_list)






def elemwise_list_bert(model, tokenizer, text_list_1, text_list_2, metric_function, method, layers):
	"""
	docstring
	"""
	assert len(text_list_1) == len(text_list_2)
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors = [_infer_document_vector_from_bert(description, model, tokenizer, method, layers) for description in descriptions]
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	assert len(list_1_vectors) == len(list_2_vectors)
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(list_1_vectors)
	return(distances_list)




def elemwise_list_ngrams(text_list_1, text_list_2, metric_function, tfidf=False, **kwargs):
	"""
	docstring	
	"""
	assert len(text_list_1) == len(text_list_2)
	descriptions = []
	descriptions.extend(text_list_1)
	descriptions.extend(text_list_2)
	all_vectors,vectorizer = strings_to_numerical_vectors(*descriptions, **kwargs)
	list_1_vectors = all_vectors[:len(text_list_1)]
	list_2_vectors = all_vectors[len(text_list_1):]
	assert len(list_1_vectors) == len(list_2_vectors)
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(list_1_vectors)
	return(distances_list)





def elemwise_list_annotations(annotations_list_1, annotations_list_2, ontology, metric_function, tfidf=False, **kwargs):
	"""
	docstring	
	"""
	assert len(annotations_list_1) == len(annotations_list_2)
	joined_term_strings = []
	all_annotations_lists = []
	all_annotations_lists.extend(annotations_list_1)
	all_annotations_lists.extend(annotations_list_2)
	for term_list in all_annotations_lists:
		term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
		term_list = flatten(term_list)
		term_list = list(set(term_list))
		joined_term_string = " ".join(term_list).strip()
		joined_term_strings.append(joined_term_string)

	all_vectors,vectorizer = strings_to_numerical_vectors(*joined_term_strings, tfidf=tfidf, **kwargs)
	list_1_vectors = all_vectors[:len(annotations_list_1)]
	list_2_vectors = all_vectors[len(annotations_list_1):]
	assert len(list_1_vectors) == len(list_2_vectors)
	vector_pairs = zip(list_1_vectors, list_2_vectors)
	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
	assert len(distances_list) == len(list_1_vectors)
	return(distances_list)























































