from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import numpy as np
import pandas as pd
import random


from oats.utils.utils import flatten
from oats.distances.distances import SquarePairwiseDistances
from oats.distances.distances import RectangularPairwiseDistances
from oats.distances._utils import _square_adjacency_matrix_to_edgelist
from oats.distances._utils import _rectangular_adjacency_matrix_to_edgelist
from oats.distances._utils import _strings_to_count_vectors
from oats.distances._utils import _strings_to_tfidf_vectors
#from oats.distances._utils import _infer_document_vector_from_bert			     # Is actually called when finding distance matrices.
#from oats.distances._utils import _infer_document_vector_from_word2vec		     # Is actually called when finding distance matrices.
#from oats.distances._utils import _infer_document_vector_from_doc2vec		     # Is actually called when finding distance matrices.
#from oats.distances._utils import _infer_document_vector_from_similarity_matrix  # Is actually called when finding distance matrices.
#from oats.distances._utils import _get_ngrams_vector						     # Not called, just a wrapper function to remember a vectorization scheme.
#from oats.distances._utils import _get_annotations_vector					     # Not called, just a wrapper function to remember a vectorization scheme.
#from oats.distances._utils import _get_topic_model_vector 					     # Not called, just a wrapper function to remember a vectorization scheme.








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












def vectorize_with_bert(text, model, tokenizer, method="sum", layers=4):
	"""
	This function uses a pretrained BERT model to infer a document level vector for a collection 
	of one or more sentences. The sentence are defined using the nltk sentence parser. This is 
	done because the BERT encoder expects either a single sentence or a pair of sentences. The
	internal representations are drawn from the last n layers as specified by the layers argument, 
	and represent a particular token but account for the context that it is in because the entire
	sentence is input simultanously. The vectors for the layers can concatentated or summed 
	together based on the method argument. The vector obtained for each token then are averaged
	together to for the document level vector.
	
	Args:
	    text (str):  Any arbitrary text string.
	    
	    model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
	    
	    tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
	    
	    method (str): A string indicating how layers for a token should be combined (concat or sum).
	    
	    layers (int): An integer saying how many layers should be used for each token.
	
	Returns:
	    numpy.Array: A numpy array which is the vector embedding for the passed in text. 
	
	Raises:
	    ValueError: The method argument has to be either 'concat' or 'sum'.
	"""

	sentences = sent_tokenize(text)
	token_vecs_cat = []
	token_vecs_sum = []

	for text in sentences:
		marked_text = "{} {} {}".format("[CLS]",text,"[SEP]")
		tokenized_text = tokenizer.tokenize(marked_text)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensor = torch.tensor([segments_ids])
		with torch.no_grad():
			encoded_layers,_ = model(tokens_tensor,segments_tensor)
		token_embeddings = torch.stack(encoded_layers, dim=0)
		token_embeddings = token_embeddings.permute(1,2,0,3)
		batch = 0
		for token in token_embeddings[batch]:
			concatenated_layer_vectors = torch.cat(tuple(token[-layers:]), dim=0)
			summed_layer_vectors = torch.sum(token[-layers:], dim=0)
			token_vecs_cat.append(np.array(concatenated_layer_vectors))
			token_vecs_sum.append(np.array(summed_layer_vectors))

	# Check to make sure atleast one token was found with an embedding to use as a the 
	# vector representation. If there wasn't found, this is because of the combination
	# of what the passed in description was, and how it was handled by either the sentence
	# tokenizing step or the BERT tokenizer methods. Handle this by generating a random
	# vector. This makes the embedding meaningless but prevents multiple instances that
	# do not have embeddings from clustering together in downstream analysis. An expected
	# layer size is hardcoded for this section based on the BERT architecture.
	expected_layer_size = 768
	if len(token_vecs_cat) == 0:
		print("no embeddings found for input text '{}', generating random vector".format(description))
		random_concat_vector = np.random.rand(expected_layer_size*layers)
		random_summed_vector = np.random.rand(expected_layer_size)
		token_vecs_cat.append(random_concat_vector)
		token_vecs_sum.append(random_summed_vector)

	# Average the vectors obtained for each token across all the sentences present in the input text.
	if method == "concat":
		embedding = np.mean(np.array(token_vecs_cat),axis=0)
	elif method == "sum":
		embedding = np.mean(np.array(token_vecs_sum),axis=0)
	else:
		raise ValueError("method argument is invalid")
	return(embedding)








def vectorize_with_word2vec(text, model, method):
	"""Generate a vector representation of a text string using Word2Vec.

	Args:
	    text (str): Any arbitrary text string.
	    
	    model (gensim.models.Word2Vec): A loaded Word2Vec model object.
	    
	    method (str): Either 'mean' or 'max', indicating how word vectors should combine to form the document vector.
	
	Returns:
	    numpy.Array: A numerical vector with length determined by the model used.
	
	Raises:
	    Error: The string for the method argument is not recognized.
	"""
	words = text.lower().split()
	words_in_model_vocab = [word for word in words if word in model.wv.vocab]
	if len(words_in_model_vocab) == 0:
		words_in_model_vocab.append(random.choice(list(model.wv.vocab)))
	stacked_vectors = np.array([model.wv[word] for word in words_in_model_vocab])
	if method == "mean":
		vector = stacked_vectors.mean(axis=0)
	elif method == "max":
		vector = stacked_vectors.max(axis=0)
	else:
		raise Error("method argument is invalid")
	return(vector)








def vectorize_with_doc2vec(text, model):
	"""Genereate a vector representation of a text string using Doc2Vec.
	
	Args:
	    text (str): Any arbitrary text string.
	    
	    model (gensim.models.Doc2Vec): A loaded Doc2Vec model object.
	
	Returns:
	    numpy.Array: A numerical vector with length determined by the model used.
	"""
	vector = model.infer_vector(text.lower().split())
	return(vector)








def vectorize_with_similarities(text, vocab_tokens, vocab_token_to_index, vocab_matrix):
	"""
	Generate a vector representation of a text string based on a word similarity matrix. The resulting vector has 
	n positions, where n is the number of words or tokens in the full vocabulary. The value at each position indicates
	the maximum similarity between that corresponding word in the vocabulary and any of the words or tokens in the
	input text string, as given by the input similarity matrix. Therefore, this is similar to an n-grams approach but
	uses the similarity between non-identical words or tokens to make the vector semantically meaningful.
	
	Args:
	    text (str): Any arbitrary text string. 

	    vocab_tokens (list of str): The words or tokens that make up the entire vocabulary.
	    
	    vocab_token_to_index (dict of str:int): Mapping between words in the vocabulary and an index in rows and columns of the matrix.
	    
	    vocab_matrix (numpy.array): A pairwise distance matrix holding the similarity values between all possible pairs of words in the vocabulary.
	
	Returns:
	    numpy.Array: A numerical vector with length equal to the size of the vocabulary.
	"""
	doc_tokens = [token for token in text.split() if token in vocab_tokens]
	vector = [max([vocab_matrix[vocab_token_to_index[vocab_token]][vocab_token_to_index[doc_token]] for doc_token in doc_tokens]) for vocab_token in vocab_tokens]
	return(vector)












def _for_new_texts_get_ngrams_vector(text, vectorizer):
	"""Not called here, just used to store a method that contains a vectorizer to be used later.
	"""
	vector = vectorizer.transform([text]).toarray()[0]
	return(vector)




def _for_new_texts_get_annotations_vector(term_list, vectorizer, ontology):
	"""Not called here, just used to store a method that contains a vectorizer to be used later.
	"""
	term_list = [ontology.subclass_dict.get(x, x) for x in term_list]
	term_list = flatten(term_list)
	term_list = list(set(term_list))
	joined_term_string = " ".join(term_list).strip()
	vector = vectorizer.transform([joined_term_string]).toarray()[0]
	return(vector)



def _for_new_texts_get_topic_model_vector(text, vectorizer, topic_model):
	"""Not called here, just used to store a method that contains a vectorizer to be used later.
	"""
	ngram_vector = vectorizer.transform([text]).toarray()[0]
	topic_vector = topic_model.transform([ngram_vector])[0]
	return(topic_vector)




















def vectorize_with_ngrams(strs, training_texts=None, tfidf=False, **kwargs):
	"""Create a vector embedding for each passed in text string.
	
	Args:
	    strs (list of str): A list of text strings that will each be translated into a numerical vector.
	    
	    training_texts (list of str, optional): If provided, this is the list of texts that will be used to the determine the vocabulary and weights for each token.
	    
	    tfidf (bool, optional): This value is false by default, set to true to use term-frequency inverse-document-frequency weighting instead of raw counts.
	    
	    **kwargs: Any other applicable keyword arguments that will be passed to the sklearn vectorization function.
	
	Returns:
	    list of numpy.Array, object: A list of the numerical vector arrays that is the same length as the input list of text strings, and the vectorizing object.
	"""
	if tfidf:
		return(_strings_to_tfidf_vectors(strs, training_texts, **kwargs))
	else:
		return(_strings_to_count_vectors(strs, training_texts, **kwargs))

















def _pairwise_square_general_case(ids_to_something, to_vector_now, to_vector_now_kwargs, to_vector_later, to_vector_later_kwargs, metric):
	"""Version that includes all the common code so it only has to appear once, and the parts that differ are passed in as functions.
	
	Args:
	    ids_to_something (TYPE): Description
	    
	    to_vector_now (TYPE): Description
	    
	    to_vector_now_kwargs (TYPE): Description
	    
	    to_vector_later (TYPE): Description
	    
	    to_vector_later_kwwargs (TYPE): Description
	    
	    metric (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""

	vectors = []
	index_in_matrix_to_id = {}
	id_to_index_in_matrix = {}
	for identifier,something in ids_to_something.items():
		vector = to_vector_now(something, **to_vector_now_kwargs)
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
		# Arguments that are always created here in this general case method.
		metric_str=metric,
		edgelist=edgelist,
		array=matrix,
		vector_dictionary=id_to_vector_dict,
		id_to_index=id_to_index_in_matrix,
		index_to_id=index_in_matrix_to_id,
		# Arguments that vary between approaches used and might be created outside the gneral case method.
		vectorizing_function=to_vector_later,
		vectorizing_function_kwargs=to_vector_later_kwargs,
		vectorizer_object=None))























def with_precomputed_vectors(ids_to_vectors, metric):
	"""Summary

	Args:
	    ids_to_vectors (TYPE): Description

	    metric (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""

	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	to_vector_function = lambda x: x
	to_vector_kwargs = {}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_vectors, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=to_vector_function,
		to_vector_later_kwargs=to_vector_kwargs,
		metric=metric))






def with_doc2vec(model, ids_to_texts, metric):
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

	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	to_vector_function = vectorize_with_doc2vec
	to_vector_kwargs = {"model":model}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=to_vector_function,
		to_vector_later_kwargs=to_vector_kwargs,
		metric=metric))





def with_word2vec(model, ids_to_texts, metric, method="mean"):
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
	

	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	to_vector_function = vectorize_with_word2vec
	to_vector_kwargs = {"model":model, "method":method}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=to_vector_function,
		to_vector_later_kwargs=to_vector_kwargs,
		metric=metric))












def with_similarities(ids_to_texts, vocab_tokens, vocab_matrix, metric):

	# Check the make sure that the vocabulary similarity matrix is the correct size for the vocabulary.
	assert vocab_matrix.shape[0] == len(vocab_tokens)
	assert vocab_matrix.shape[1] == len(vocab_tokens)
	vocab_token_to_index = {vocab_token:i for i,vocab_token in enumerate(vocab_tokens)}



	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	to_vector_function = vectorize_with_similarities
	to_vector_kwargs = {"vocab_tokens":vocab_tokens, "vocab_token_to_index":vocab_token_to_index, "vocab_matrix":vocab_matrix}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=to_vector_function,
		to_vector_later_kwargs=to_vector_kwargs,
		metric=metric))








def with_bert(model, tokenizer, ids_to_texts, metric, method, layers):
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


	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	to_vector_function = vectorize_with_bert
	to_vector_kwargs = {"model":model, "tokenizer":tokenizer, "method":method, "layers":layers}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=to_vector_function,
		to_vector_later_kwargs=to_vector_kwargs,
		metric=metric))






def with_ngrams(ids_to_texts, metric, training_texts=None, tfidf=False, **kwargs):
	"""
	Find distance between strings of text in some input data using n-grams. Note that only 
	very simple preprocessing is done after this point (splitting on whitespace only) so 
	all processing of the text necessary should be done prio to passing to this function.
	
	Args:
	    ids_to_texts (dict): A mapping between IDs and strings of text.
	    
	    metric (str): A string indicating which distance metric should be used (e.g., cosine).
	    
	    training_texts (None, optional): Description
	    
	    tfidf (bool, optional): Whether to use TFIDF weighting or' not.
	    
	    **kwargs: All the keyword arguments that can be passed to sklearn.feature_extraction.CountVectorizer().
	
	Returns:
	    oats.pairwise.SquarePairwiseDistances: Distance matrix and accompanying information.
	"""



	# Because we can generate all the vectors at once with n-grams, we actually find the explicit mapping here and then pass it as a dummy function.
	descriptions = ids_to_texts.values()
	vectors,vectorizer = vectorize_with_ngrams(descriptions, training_texts, tfidf=tfidf, **kwargs)
	description_to_vector_mapping = {text:vector for text,vector in zip(descriptions,vectors)}
	

	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	# Note that in this case the functions and arguments for getting the distance matrix now and embedding new text in the future are different.
	# Now we use the vectors that were generated from fitting the vocabulary directly from this data, and that saves a vectorizer object.
	# Later, we directly use that vectorizer object instead to find the vectors for new instances of text.

	# TODO what about saving the vectorizer object?
	to_vector_function = lambda text, mapping=description_to_vector_mapping: mapping[text]
	to_vector_kwargs = {}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=_for_new_texts_get_ngrams_vector,
		to_vector_later_kwargs={"vectorizer":vectorizer},
		metric=metric))






def with_topic_model(ids_to_texts, metric, training_texts=None, seed=124134, num_topics=10, algorithm="lda", **kwargs):
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


	# The list of descriptions that are present in this dataset.
	descriptions = ids_to_texts.values()


	# Apply distance metric over all the vectors to yield a matrix.
	if training_texts is not None:
		vectorizer.fit(training_texts)
	else:
		vectorizer.fit(descriptions)


	# Find the n-grams vector representations for each text instance, and use that set to fit the topic model.
	ngram_vectors = vectorizer.transform(descriptions).toarray()
	topic_vectors = model.fit_transform(ngram_vectors)
	description_to_vector_mapping = {text:vector for text,vector in zip(descriptions,topic_vectors)}
	

	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.
	# Note that in this case the functions and arguments for getting the distance matrix now and embedding new text in the future are different.
	# Now we use the vectors that were generated from fitting the vocabulary directly from this data, and that saves a vectorizer object.
	# Later, we directly use that vectorizer object instead to find the vectors for new instances of text.
	to_vector_function = lambda text, mapping=description_to_vector_mapping: mapping[text]
	to_vector_kwargs = {}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_texts, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
	 	to_vector_later = _for_new_texts_get_topic_model_vector,
	 	to_vector_later_kwargs = {"vectorizer":vectorizer, "topic_model":model},
		metric=metric))







def with_annotations(ids_to_annotations, ontology, metric, tfidf=False, **kwargs):
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


	# Generate the vector representations of each set of annotations by first inheriting terms then converting to strings.
	ids_to_term_lists = {i:list(set(flatten([ontology.inherited(term_id) for term_id in term_list]))) for i,term_list in ids_to_annotations.items()}
	ids_to_joined_term_strings = {i:" ".join(term_list).strip() for i,term_list in ids_to_term_lists.items()}
	joined_term_strings_list = ids_to_joined_term_strings.values()
	vectors, vectorizer = vectorize_with_ngrams(joined_term_strings_list, tfidf=tfidf, **kwargs)
	joined_term_strings_to_vector_mapping = {term_list_string:vector for term_list_string,vector in zip(joined_term_strings_list,vectors)}


	# Send the relevant functions and arguments to the general case method for generating the distance matrix object.s
	to_vector_function = lambda term_list_string, mapping=joined_term_strings_to_vector_mapping: mapping[term_list_string]
	to_vector_kwargs = {}
	return(_pairwise_square_general_case(
		ids_to_something=ids_to_joined_term_strings, 
		to_vector_now=to_vector_function, 
		to_vector_now_kwargs=to_vector_kwargs, 
		to_vector_later=_for_new_texts_get_annotations_vector,
		to_vector_later_kwargs={"vectorizer":vectorizer, "ontology":ontology},
		metric=metric))



































# def pairwise_rectangular_precomputed_vectors(ids_to_vectors_1, ids_to_vectors_2, metric):
# 	"""docstring
# 	"""
# 	vectors = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0	
# 	for identifier,vector in ids_to_vectors_1.items():
# 		vectors.append(vector)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,vector in ids_to_vectors_2.items():
# 		vectors.append(vector)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	all_vectors = vectors
# 	row_vectors = all_vectors[:len(ids_to_vectors_1)]
# 	col_vectors = all_vectors[len(ids_to_vectors_1):]
# 	row_id_to_vector_dict = ids_to_vectors_1
# 	col_id_to_vector_dict = ids_to_vectors_2
# 	matrix = cdist(row_vectors, col_vectors, metric)
# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)



# 	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = None,		
# 		vectorizing_function_kwargs = None,		
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = None,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))










# def pairwise_rectangular_doc2vec(model, ids_to_texts_1, ids_to_texts_2, metric):
# 	"""docstring
# 	"""
# 	vectors = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0	
# 	for identifier,description in ids_to_texts_1.items():
# 		inferred_vector = _infer_document_vector_from_doc2vec(description, model)
# 		vectors.append(inferred_vector)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,description in ids_to_texts_2.items():
# 		inferred_vector = _infer_document_vector_from_doc2vec(description, model)
# 		vectors.append(inferred_vector)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	all_vectors = vectors
# 	row_vectors = all_vectors[:len(ids_to_texts_1)]
# 	col_vectors = all_vectors[len(ids_to_texts_1):]
# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)
# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)



# 	# Create and return a SquarePairwiseDistances object containing the edgelist, matrix, and dictionaries.
# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _infer_document_vector_from_doc2vec,		
# 		vectorizing_function_kwargs = {"model":model},		
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = None,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))








# def pairwise_rectangular_word2vec(model, ids_to_texts_1, ids_to_texts_2, metric, method="mean"):
# 	"""
# 	docstring
# 	"""
# 	vectors = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0	
# 	for identifier,description in ids_to_texts_1.items():
# 		vector = _infer_document_vector_from_word2vec(description, model, method)
# 		vectors.append(vector)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,description in ids_to_texts_2.items():
# 		vector = _infer_document_vector_from_word2vec(description, model, method)
# 		vectors.append(vector)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	all_vectors = vectors
# 	row_vectors = all_vectors[:len(ids_to_texts_1)]
# 	col_vectors = all_vectors[len(ids_to_texts_1):]
# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)
# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	

# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _infer_document_vector_from_word2vec,
# 		vectorizing_function_kwargs = {"model":model, "method":method},
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = None,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))










# def pairwise_rectangular_bert(model, tokenizer, ids_to_texts_1, ids_to_texts_2, metric, method, layers):
# 	"""
# 	docstring
# 	"""
# 	vectors = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0	
# 	for identifier,description in ids_to_texts_1.items():
# 		inferred_vector = _infer_document_vector_from_bert(description, model, tokenizer, method, layers)
# 		vectors.append(inferred_vector)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,description in ids_to_texts_2.items():
# 		inferred_vector = _infer_document_vector_from_bert(description, model, tokenizer, method, layers)
# 		vectors.append(inferred_vector)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	all_vectors = vectors
# 	row_vectors = all_vectors[:len(ids_to_texts_1)]
# 	col_vectors = all_vectors[len(ids_to_texts_1):]
# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)
# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	

# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _infer_document_vector_from_bert,
# 		vectorizing_function_kwargs = {"model":model, "tokenizer":tokenizer, "method":method, "layers":layers},
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = None,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))





# def pairwise_rectangular_ngrams(ids_to_texts_1, ids_to_texts_2, metric, tfidf=False, **kwargs):
# 	"""
# 	docstring
# 	"""
# 	descriptions = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0
# 	for identifier,description in ids_to_texts_1.items():
# 		descriptions.append(description)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,description in ids_to_texts_2.items():
# 		descriptions.append(description)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	all_vectors,vectorizer = strings_to_numerical_vectors(descriptions, tfidf=tfidf, **kwargs)
# 	row_vectors = all_vectors[:len(ids_to_texts_1)]
# 	col_vectors = all_vectors[len(ids_to_texts_1):]

# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)

# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	
# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _get_ngrams_vector,
# 		vectorizing_function_kwargs = {"countvectorizer":vectorizer},
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = vectorizer,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))










# def pairwise_rectangular_topic_model(ids_to_texts_1, ids_to_texts_2, metric, seed=124134, num_topics=10, algorithm="LDA", **kwargs):
# 	"""
# 	docstring
# 	"""


# 	# Fitting the topic model using the provided parameters and this dataset of text descriptions.
# 	vectorizer = TfidfVectorizer(**kwargs)
# 	if algorithm.lower() == "lda":
# 		model = LDA(n_components=num_topics, random_state=seed)
# 	elif algorithm.lower() == "nmf":
# 		model = NMF(n_components=num_topics, random_state=seed)
# 	else:
# 		raise ValueError("algorithm argument is invalid")


# 	descriptions = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0
# 	for identifier,description in ids_to_texts_1.items():
# 		descriptions.append(description)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,description in ids_to_texts_2.items():
# 		descriptions.append(description)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	# Apply distance metric over all the vectors to yield a matrix.
# 	ngram_vectors = vectorizer.fit_transform(descriptions).toarray()
# 	topic_vectors = model.fit_transform(ngram_vectors)
# 	all_vectors = topic_vectors
# 	row_vectors = all_vectors[:len(ids_to_texts_1)]
# 	col_vectors = all_vectors[len(ids_to_texts_1):]

# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)

# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)
	
# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _get_topic_model_vector,
# 		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "topic_model":model},
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = vectorizer,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))





















# def pairwise_rectangular_annotations(ids_to_annotations_1, ids_to_annotations_2, ontology, metric, tfidf=False, **kwargs):
# 	"""
# 	docstring
# 	"""
# 	joined_term_strings = []
# 	row_index_in_matrix_to_id = {}
# 	col_index_in_matrix_to_id = {}
# 	id_to_row_index_in_matrix = {}
# 	id_to_col_index_in_matrix = {}

# 	row_in_matrix = 0
# 	for identifier,term_list in ids_to_annotations_1.items():

# 		print(term_list)

# 		term_list = [ontology.inherited(x) for x in term_list]

# 		print(term_list)

# 		term_list = flatten(term_list)

# 		print(term_list)
# 		term_list = list(set(term_list))

# 		print(term_list)
# 		joined_term_string = " ".join(term_list).strip()
# 		joined_term_strings.append(joined_term_string)
# 		row_index_in_matrix_to_id[row_in_matrix] = identifier
# 		id_to_row_index_in_matrix[identifier] = row_in_matrix 
# 		row_in_matrix = row_in_matrix+1

# 	col_in_matrix = 0
# 	for identifier,term_list in ids_to_annotations_2.items():
# 		term_list = [ontology.inherited(x) for x in term_list]
# 		term_list = flatten(term_list)
# 		term_list = list(set(term_list))
# 		joined_term_string = " ".join(term_list).strip()
# 		joined_term_strings.append(joined_term_string)
# 		col_index_in_matrix_to_id[col_in_matrix] = identifier
# 		id_to_col_index_in_matrix[identifier] = col_in_matrix 
# 		col_in_matrix = col_in_matrix+1

# 	# Find all the pairwise values for the distance matrix.
# 	all_vectors,vectorizer = strings_to_numerical_vectors(joined_term_strings, tfidf=tfidf, **kwargs)
# 	row_vectors = all_vectors[:len(ids_to_annotations_1)]
# 	col_vectors = all_vectors[len(ids_to_annotations_1):]

# 	row_id_to_vector_dict = {row_index_in_matrix_to_id[i]:vector for i,vector in enumerate(row_vectors)}
# 	col_id_to_vector_dict = {col_index_in_matrix_to_id[i]:vector for i,vector in enumerate(col_vectors)}
# 	matrix = cdist(row_vectors, col_vectors, metric)	
# 	edgelist = _rectangular_adjacency_matrix_to_edgelist(matrix, row_index_in_matrix_to_id, col_index_in_matrix_to_id)


# 	return(RectangularPairwiseDistances(
# 		metric_str = metric,
# 		vectorizing_function = _get_annotations_vector,
# 		vectorizing_function_kwargs = {"countvectorizer":vectorizer, "ontology":ontology},
# 		edgelist = edgelist,
# 		row_vector_dictionary = row_id_to_vector_dict,
# 		col_vector_dictionary = col_id_to_vector_dict,
# 		vectorizer_object = vectorizer,
# 		id_to_row_index=id_to_row_index_in_matrix, 
# 		id_to_col_index=id_to_col_index_in_matrix,
# 		row_index_to_id=row_index_in_matrix_to_id, 
# 		col_index_to_id=col_index_in_matrix_to_id,
# 		array=matrix))























# def elemwise_list_precomputed_vectors(vector_list_1, vector_list_2, metric_function):
# 	"""
# 	docstring
# 	"""
# 	assert len(vector_list_1) == len(vector_list_2)
# 	vector_pairs = zip(vector_list_1, vector_list_2)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(vector_list_1)
# 	return(distances_list)







# def elemwise_list_doc2vec(model, text_list_1, text_list_2, metric_function):
# 	"""
# 	docstring
# 	"""
# 	assert len(text_list_1) == len(text_list_2)
# 	descriptions = []
# 	descriptions.extend(text_list_1)
# 	descriptions.extend(text_list_2)
# 	all_vectors = [_infer_document_vector_from_doc2vec(description, model) for description in descriptions]
# 	list_1_vectors = all_vectors[:len(text_list_1)]
# 	list_2_vectors = all_vectors[len(text_list_1):]
# 	assert len(list_1_vectors) == len(list_2_vectors)
# 	vector_pairs = zip(list_1_vectors, list_2_vectors)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(list_1_vectors)
# 	return(distances_list)




# def elemwise_list_word2vec(model, text_list_1, text_list_2, metric_function, method="mean"):
# 	"""
# 	docstring
# 	"""
# 	assert len(text_list_1) == len(text_list_2)
# 	descriptions = []
# 	descriptions.extend(text_list_1)
# 	descriptions.extend(text_list_2)
# 	all_vectors = []
# 	all_vectors = [_infer_document_vector_from_word2vec(description, model, method) for description in descriptions]
# 	list_1_vectors = all_vectors[:len(text_list_1)]
# 	list_2_vectors = all_vectors[len(text_list_1):]
# 	assert len(list_1_vectors) == len(list_2_vectors)
# 	vector_pairs = zip(list_1_vectors, list_2_vectors)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(list_1_vectors)
# 	return(distances_list)






# def elemwise_list_bert(model, tokenizer, text_list_1, text_list_2, metric_function, method, layers):
# 	"""
# 	docstring
# 	"""
# 	assert len(text_list_1) == len(text_list_2)
# 	descriptions = []
# 	descriptions.extend(text_list_1)
# 	descriptions.extend(text_list_2)
# 	all_vectors = [_infer_document_vector_from_bert(description, model, tokenizer, method, layers) for description in descriptions]
# 	list_1_vectors = all_vectors[:len(text_list_1)]
# 	list_2_vectors = all_vectors[len(text_list_1):]
# 	assert len(list_1_vectors) == len(list_2_vectors)
# 	vector_pairs = zip(list_1_vectors, list_2_vectors)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(list_1_vectors)
# 	return(distances_list)




# def elemwise_list_ngrams(text_list_1, text_list_2, metric_function, tfidf=False, **kwargs):
# 	"""
# 	docstring	
# 	"""
# 	assert len(text_list_1) == len(text_list_2)
# 	descriptions = []
# 	descriptions.extend(text_list_1)
# 	descriptions.extend(text_list_2)
# 	all_vectors,vectorizer = strings_to_numerical_vectors(descriptions, **kwargs)
# 	list_1_vectors = all_vectors[:len(text_list_1)]
# 	list_2_vectors = all_vectors[len(text_list_1):]
# 	assert len(list_1_vectors) == len(list_2_vectors)
# 	vector_pairs = zip(list_1_vectors, list_2_vectors)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(list_1_vectors)
# 	return(distances_list)





# def elemwise_list_annotations(annotations_list_1, annotations_list_2, ontology, metric_function, tfidf=False, **kwargs):
# 	"""
# 	docstring	
# 	"""
# 	assert len(annotations_list_1) == len(annotations_list_2)
# 	joined_term_strings = []
# 	all_annotations_lists = []
# 	all_annotations_lists.extend(annotations_list_1)
# 	all_annotations_lists.extend(annotations_list_2)
# 	for term_list in all_annotations_lists:
# 		term_list = [ontology.inherited(x) for x in term_list]
# 		term_list = flatten(term_list)
# 		term_list = list(set(term_list))
# 		joined_term_string = " ".join(term_list).strip()
# 		joined_term_strings.append(joined_term_string)

# 	all_vectors,vectorizer = strings_to_numerical_vectors(joined_term_strings, tfidf=tfidf, **kwargs)
# 	list_1_vectors = all_vectors[:len(annotations_list_1)]
# 	list_2_vectors = all_vectors[len(annotations_list_1):]
# 	assert len(list_1_vectors) == len(list_2_vectors)
# 	vector_pairs = zip(list_1_vectors, list_2_vectors)
# 	distances_list = [metric_function(vector_pair[0],vector_pair[1]) for vector_pair in vector_pairs]
# 	assert len(distances_list) == len(list_1_vectors)
# 	return(distances_list)























































