import pytest
import sys
import pandas as pd
import numpy as np
import json
sys.path.append("../oats")
import oats








@pytest.fixture
def doc2vec_model():
	import gensim
	doc2vec_wikipedia_filename = "/Users/irbraun/phenologs-with-oats/models/enwiki_dbow/doc2vec.bin"     
	doc2vec_wiki_model = gensim.models.Doc2Vec.load(doc2vec_wikipedia_filename)
	return(doc2vec_wiki_model)


@pytest.fixture
def word2vec_model():
	import gensim
	word2vec_model_filename = "/Users/irbraun/phenologs-with-oats/models/wiki_sg/word2vec.bin"    
	word2vec_model = gensim.models.Word2Vec.load(word2vec_model_filename)
	return(word2vec_model)


@pytest.fixture
def bert_model():
	from pytorch_pretrained_bert import BertModel
	bert_model_base = BertModel.from_pretrained('bert-base-uncased')
	return(bert_model_base)


@pytest.fixture
def bert_tokenizer():
	from pytorch_pretrained_bert import BertTokenizer
	bert_tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
	return(bert_tokenizer_base)


@pytest.fixture
def ontology():
	from oats.annotation.ontology import Ontology
	ontology_filename = "tests/large/po.obo"   
	ontology = Ontology(ontology_filename)
	return(ontology)










with open("tests/data/small_dataset.json") as f:
  data = json.load(f)








# @pytest.mark.somethingnew
# def test_new_stuff(word2vec_model):




# 	# something here
# 	a_strings = data["data_group_1"]["descriptions_dictionary"]
# 	tokens = set()
# 	for s in a_strings.values():
# 		tokens.update(s.split())
# 	vocab = list(tokens)


# 	import itertools

# 	indices = range(len(vocab))
# 	array = np.zeros((len(vocab),len(vocab)))
# 	for i,j in itertools.product(indices,indices):
# 		array[i][j] = word2vec_model.similarity(vocab[i],vocab[j])

		

# 	from oats.distances.pairwise import pairwise_square_something_from_paper
# 	g = pairwise_square_something_from_paper(ids_to_texts=a_strings, vocab_tokens=vocab, vocab_matrix=array, metric="cosine")











# @pytest.mark.simple
# def test_get_all_rectanglular_distance_matrices(word2vec_model, doc2vec_model, bert_model, bert_tokenizer, ontology):
# 	"""Making sure the methods to generate distance matrices from two sets of text instances work in the simplest cases.
	
# 	Args:
# 	    word2vec_model (TYPE): Description
# 	    doc2vec_model (TYPE): Description
# 	    bert_model (TYPE): Description
# 	    bert_tokenizer (TYPE): Description
# 	    ontology (TYPE): Description
# 	    fit_vectorizer (TYPE): Description
# 	    unfit_lda_topic_model (TYPE): Description
# 	    unfit_nmf_topic_model (TYPE): Description
# 	"""
# 	from oats.distances.pairwise import with_precomputed_vectors
# 	from oats.distances.pairwise import with_ngrams
# 	from oats.distances.pairwise import pairwise_rectangular_word2vec
# 	from oats.distances.pairwise import pairwise_rectangular_doc2vec
# 	from oats.distances.pairwise import pairwise_rectangular_bert
# 	from oats.distances.pairwise import pairwise_rectangular_annotations
# 	from oats.distances.pairwise import pairwise_rectangular_topic_model


# 	# Reading data from the json file.
# 	a_vectors = data["data_group_1"]["vectors_dictionary"]
# 	a_terms = data["data_group_1"]["annotations_dictionary"]
# 	a_strings = data["data_group_1"]["descriptions_dictionary"]
# 	b_vectors = data["data_group_2"]["vectors_dictionary"]
# 	b_terms = data["data_group_2"]["annotations_dictionary"]
# 	b_strings = data["data_group_2"]["descriptions_dictionary"]

# 	# Running all the pairwise distance matrix functions with the simplest cases of all arguments.
# 	# This test is just for catching major problems not edge cases.
# 	# The only assertion statements run are the ones inside of all of these methods.
# 	g = pairwise_rectangular_precomputed_vectors(ids_to_vectors_1=data["data_group_1"]["vectors_dictionary"], ids_to_vectors_2=b_vectors, metric="euclidean")
# 	g = pairwise_rectangular_ngrams(ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean")
# 	g = pairwise_rectangular_word2vec(word2vec_model, ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean")
# 	g = pairwise_rectangular_doc2vec(doc2vec_model, ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean")
# 	g = pairwise_rectangular_bert(bert_model, bert_tokenizer, ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean", method="concat", layers=4)
# 	g = pairwise_rectangular_annotations(ids_to_annotations_1=a_terms, ids_to_annotations_2=b_terms, ontology=ontology, metric="jaccard")
# 	g = pairwise_rectangular_topic_model(ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean", num_topics=4, algorithm="lda")
# 	g = pairwise_rectangular_topic_model(ids_to_texts_1=a_strings, ids_to_texts_2=b_strings, metric="euclidean", num_topics=4, algorithm="nmf")



# TODO fix all these tests to use integer IDs not strings, that's not how they're used everywhere else, should enforce it.


# @pytest.mark.quick
# def test_something():
# 	from oats.distances.pairwise import pairwise_square_ngrams

# 	b_strings = data["data_group_2"]["descriptions_dictionary"]

# 	g = pairwise_square_ngrams(ids_to_texts=b_strings, metric="cosine")

# 	print(g.ids)

# 	# TODO fix all these tests to use integer IDs not strings, that's not how they're used everywhere else, should enforce it.

# 	print(g.get_distances("some words here"))

# 	print(g.get_distance('1','2'))
# 	print(g.get_distance('2','1'))





@pytest.mark.aaa
def test_get_all_square_distance_matrices(word2vec_model, doc2vec_model, bert_model, bert_tokenizer, ontology):
	"""Making sure the methods to generate distance matrices from one set of text instances work in the simplest cases.
	
	Args:
	    word2vec_model (TYPE): Description
	    doc2vec_model (TYPE): Description
	    bert_model (TYPE): Description
	    bert_tokenizer (TYPE): Descriptions
	    ontology (TYPE): Description
	    fit_vectorizer (TYPE): Description
	    unfit_lda_topic_model (TYPE): Description
	    unfit_nmf_topic_model (TYPE): Description
	"""
	from oats.distances.pairwise import with_precomputed_vectors
	from oats.distances.pairwise import with_ngrams
	from oats.distances.pairwise import with_word2vec
	from oats.distances.pairwise import with_doc2vec
	from oats.distances.pairwise import with_bert
	from oats.distances.pairwise import with_annotations
	from oats.distances.pairwise import with_topic_model


	# Reading data from the json file.
	b_vectors = data["data_group_2"]["vectors_dictionary"]
	b_terms = data["data_group_2"]["annotations_dictionary"]
	b_strings = data["data_group_2"]["descriptions_dictionary"]

	# Running all the pairwise distance matrix functions with the simplest cases of all arguments.
	# This test is just for catching major problems not edge cases.
	# The only assertion statements run are the ones inside of all of these methods.
	g = with_precomputed_vectors(ids_to_vectors=b_vectors, metric="euclidean")
	g = with_ngrams(ids_to_texts=b_strings, metric="euclidean")
	g = with_word2vec(ids_to_texts=b_strings, model=word2vec_model, metric="euclidean")
	g = with_doc2vec(b_strings, doc2vec_model, metric="euclidean")
	g = with_bert(b_strings, bert_model, bert_tokenizer, metric="euclidean", method="concat", layers=4)
	g = with_annotations(ids_to_annotations=b_terms, ontology=ontology, metric="jaccard")
	g = with_topic_model(ids_to_texts=b_strings, metric="euclidean", num_topics=4, algorithm="lda")
	g = with_topic_model(ids_to_texts=b_strings, metric="euclidean", num_topics=4, algorithm="nmf")


	print(g.edgelist)
	print(g.array)
	print(g.vector_dictionary)










# @pytest.mark.simple
# def test_get_all_distance_lists(word2vec_model, doc2vec_model, bert_model, bert_tokenizer, ontology):
# 	"""Making sure the methods to get a list of element-wise distances between two lists of texts work in the simplest cases.
	
# 	Args:
# 	    word2vec_model (TYPE): Description
# 	    doc2vec_model (TYPE): Description
# 	    bert_model (TYPE): Description
# 	    bert_tokenizer (TYPE): Description
# 	    ontology (TYPE): Description
# 	"""
# 	from oats.distances.pairwise import elemwise_list_precomputed_vectors
# 	from oats.distances.pairwise import elemwise_list_ngrams
# 	from oats.distances.pairwise import elemwise_list_word2vec
# 	from oats.distances.pairwise import elemwise_list_doc2vec
# 	from oats.distances.pairwise import elemwise_list_bert
# 	from oats.distances.pairwise import elemwise_list_annotations
	
# 	from scipy.spatial.distance import euclidean, cosine, jaccard
	

# 	# Reading data from the json file.
# 	b_vectors = data["data_group_2"]["vectors_dictionary"].values()
# 	b_terms = data["data_group_2"]["annotations_dictionary"].values()
# 	b_strings = data["data_group_2"]["descriptions_dictionary"].values()


# 	# Running all the element-wise distance functions with the simplest cases of all arguments.
# 	# This test is just for catching major problems not edge cases.
# 	# The only assertion statements run are the ones inside of all of these methods.
# 	g = elemwise_list_precomputed_vectors(vector_list_1=b_vectors, vector_list_2=b_vectors, metric_function=euclidean)
# 	g = elemwise_list_ngrams(text_list_1=b_strings, text_list_2=b_strings, metric_function=euclidean)
# 	g = elemwise_list_word2vec(model=word2vec_model, text_list_1=b_strings, text_list_2=b_strings, metric_function=euclidean)
# 	g = elemwise_list_doc2vec(model=doc2vec_model, text_list_1=b_strings, text_list_2=b_strings, metric_function=euclidean)
# 	g = elemwise_list_bert(model=bert_model, tokenizer=bert_tokenizer, text_list_1=b_strings, text_list_2=b_strings, metric_function=euclidean, method="concat", layers=4)
# 	g = elemwise_list_annotations(annotations_list_1=b_terms, annotations_list_2=b_terms, ontology=ontology, metric_function=jaccard)















"""
@pytest.mark.parametrize("data, metric", [
    (b, "euclidean"),   
    (a, "euclidean"),   
    (a, "jaccard"),       
])
def test_pairwise_square_ngrams(data, metric):
	from oats.graphs.pairwise import pairwise_square_ngrams
	g = pairwise_square_ngrams(ids_to_texts=data, metric=metric)






@pytest.mark.parametrize("data, metric", [
    (b, "euclidean"),   
    (b, "euclidean"),   
    (b, "jaccard"),       
])
def test_pairwise_square_doc2vec(data, metric, doc2vec_model):
	from oats.graphs.pairwise import pairwise_square_doc2vec
	g = pairwise_square_doc2vec(model=doc2vec_model, ids_to_texts=data, metric=metric)
"""





