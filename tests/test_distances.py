import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats







a = {1:"some words here", 2:"some other words"}
b = {3:"some words here", 4:"some other words here", 5:"and something"}








@pytest.fixture
def doc2vec_model():
	import gensim
	doc2vec_wikipedia_filename = "../phenologs-with-oats/gensim/enwiki_dbow/doc2vec.bin"     
	doc2vec_wiki_model = gensim.models.Doc2Vec.load(doc2vec_wikipedia_filename)
	return(doc2vec_wiki_model)


@pytest.fixture
def word2vec_model():
	import gensim
	word2vec_model_filename = "../phenologs-with-oats/gensim/wiki_sg/word2vec.bin"    
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










@pytest.mark.slow
def test_get_all_rectanglular_distance_matrices(word2vec_model, doc2vec_model, bert_model, bert_tokenizer):
	from oats.graphs.pairwise import pairwise_rectangular_ngrams
	from oats.graphs.pairwise import pairwise_rectangular_word2vec
	from oats.graphs.pairwise import pairwise_rectangular_doc2vec
	from oats.graphs.pairwise import pairwise_rectangular_bert
	g = pairwise_rectangular_ngrams(ids_to_texts_1=a, ids_to_texts_2=b, metric="euclidean")
	g = pairwise_rectangular_word2vec(word2vec_model, ids_to_texts_1=a, ids_to_texts_2=b, metric="euclidean")
	g = pairwise_rectangular_doc2vec(doc2vec_model, ids_to_texts_1=a, ids_to_texts_2=b, metric="euclidean")
	g = pairwise_rectangular_bert(bert_model, bert_tokenizer, ids_to_texts_1=a, ids_to_texts_2=b, metric="euclidean", method="concat", layers=4)








@pytest.mark.slow
def test_get_all_square_distance_matrices(word2vec_model, doc2vec_model, bert_model, bert_tokenizer):
	from oats.graphs.pairwise import pairwise_square_ngrams
	from oats.graphs.pairwise import pairwise_square_word2vec
	from oats.graphs.pairwise import pairwise_square_doc2vec
	from oats.graphs.pairwise import pairwise_square_bert
	g = pairwise_square_ngrams(ids_to_texts=b, metric="euclidean")
	g = pairwise_square_word2vec(word2vec_model, ids_to_texts=b, metric="euclidean")
	g = pairwise_square_doc2vec(doc2vec_model, ids_to_texts=b, metric="euclidean")
	g = pairwise_square_bert(bert_model, bert_tokenizer, ids_to_texts=b, metric="euclidean", method="concat", layers=4)











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






