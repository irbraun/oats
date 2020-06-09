import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats





# From the test ontology file:
#! This ontology is set up to contain two branches below the root.
#!
#!
#!                                                 --> 2       --> 5
#! The first branch of the ontology goes like:   1       --> 4 
#!                                                 --> 3       --> 6
#!
#!
#! The second branch of the ontology goes like:  1 --> 7 --> 8 --> 9 
#!
#!
#! This is done to model different aspects of a DAG, such as one term having
#! two different parents but those parents then being subclasses of a single
#! common term. The second branch just includes a linear set of terms that
#! subclasses of one another going back up to the root term. 




@pytest.fixture
def ontology():
	from oats.annotation.ontology import Ontology
	ontology_filename = "tests/data/test_ontology.obo"   
	ontology = Ontology(ontology_filename)
	return(ontology)





@pytest.mark.fast
def test_ontology_term_depths(ontology):
	"""Are the depths of each term what are expected?
	"""
	assert ontology.depth("TO:0000001") == 0
	assert ontology.depth("TO:0000002") == 1
	assert ontology.depth("TO:0000003") == 1
	assert ontology.depth("TO:0000004") == 2
	assert ontology.depth("TO:0000005") == 3
	assert ontology.depth("TO:0000006") == 3
	assert ontology.depth("TO:0000007") == 1
	assert ontology.depth("TO:0000008") == 2
	assert ontology.depth("TO:0000009") == 3



@pytest.mark.fast
def test_ontology_term_graph_based_information_content(ontology):
	"""Is the information content calculated from the graph structure what is expected?
	"""
	assert ontology.ic("TO:0000001") == 0.000
	assert ontology.ic("TO:0000002") == 0.3690702464285426
	assert ontology.ic("TO:0000003") == 0.3690702464285426
	assert ontology.ic("TO:0000004") == 1.000
	assert ontology.ic("TO:0000005") == 3.000
	assert ontology.ic("TO:0000006") == 3.000
	assert ontology.ic("TO:0000007") == 0.500
	assert ontology.ic("TO:0000008") == 1.3690702464285427
	assert ontology.ic("TO:0000009") == 3.000




@pytest.mark.fast
def test_ontology_term_inheritance(ontology):
	"""Is the number of inherited terms of each term in the graph as expected?
	"""
	assert len(ontology.inherited("TO:0000001")) == 1
	assert len(ontology.inherited("TO:0000002")) == 2
	assert len(ontology.inherited("TO:0000003")) == 2
	assert len(ontology.inherited("TO:0000004")) == 4
	assert len(ontology.inherited("TO:0000005")) == 5
	assert len(ontology.inherited("TO:0000006")) == 5
	assert len(ontology.inherited("TO:0000007")) == 2
	assert len(ontology.inherited("TO:0000008")) == 3
	assert len(ontology.inherited("TO:0000009")) == 4

	assert len(ontology.inherited(["TO:0000002","TO:0000003"])) == 3
	assert len(ontology.inherited(["TO:0000009","TO:0000005"])) == 8
	assert len(ontology.inherited(["TO:0000004","TO:0000003"])) == 4
	assert len(ontology.inherited(["TO:0000002"])) == 2
	assert len(ontology.inherited([])) == 0






@pytest.mark.fast
def test_ontology_ic_similarity(ontology):
	"""Is the information content of the most informative common ancestor term as expected for these lists of terms?
	"""
	assert ontology.similarity_ic(["TO:0000001"],["TO:0000002"], inherited=False) == 0
	assert ontology.similarity_ic(["TO:0000001"],["TO:0000003"], inherited=False) == 0
	assert ontology.similarity_ic(["TO:0000002"],["TO:0000003"], inherited=False) == 0
	assert ontology.similarity_ic(["TO:0000003"],["TO:0000005"], inherited=False) == 0.3690702464285426
	assert ontology.similarity_ic(["TO:0000007"],["TO:0000008"], inherited=False) == 0.5
	assert ontology.similarity_ic(["TO:0000005"],["TO:0000009"], inherited=False) == 0

	assert ontology.similarity_ic(["TO:0000001"],["TO:0000002","TO:0000001"], inherited=False) == 0
	assert ontology.similarity_ic(["TO:0000003"],["TO:0000001","TO:0000009"], inherited=False) == 0
	assert ontology.similarity_ic(["TO:0000002"],["TO:0000003","TO:0000002"], inherited=False) == 0.3690702464285426
	assert ontology.similarity_ic(["TO:0000003"],["TO:0000005","TO:0000002"], inherited=False) == 0.3690702464285426
	assert ontology.similarity_ic(["TO:0000008"],["TO:0000008","TO:0000007"], inherited=False) == 1.3690702464285427
	assert ontology.similarity_ic(["TO:0000005"],["TO:0000009","TO:0000002"], inherited=False) == 0.3690702464285426


@pytest.mark.fast
def test_ontology_similarity_jaccard(ontology):
	"""Is the Jaccard similarity between the given lists of terms as expected?
	"""
	assert ontology.similarity_jaccard(["TO:0000001"],["TO:0000002"], inherited=False) == 1/2
	assert ontology.similarity_jaccard(["TO:0000001"],["TO:0000003"], inherited=False) == 1/2
	assert ontology.similarity_jaccard(["TO:0000002"],["TO:0000003"], inherited=False) == 1/3
	assert ontology.similarity_jaccard(["TO:0000003"],["TO:0000005"], inherited=False) == 2/5
	assert ontology.similarity_jaccard(["TO:0000007"],["TO:0000008"], inherited=False) == 2/3
	assert ontology.similarity_jaccard(["TO:0000005"],["TO:0000009"], inherited=False) == 1/8

	assert ontology.similarity_jaccard(["TO:0000001"],["TO:0000002","TO:0000001"], inherited=False) == 1/2
	assert ontology.similarity_jaccard(["TO:0000003"],["TO:0000001","TO:0000009"], inherited=False) == 1/5
	assert ontology.similarity_jaccard(["TO:0000002"],["TO:0000003","TO:0000002"], inherited=False) == 2/3 
	assert ontology.similarity_jaccard(["TO:0000003"],["TO:0000005","TO:0000002"], inherited=False) == 2/5
	assert ontology.similarity_jaccard(["TO:0000008"],["TO:0000008","TO:0000007"], inherited=False) == 3/3
	assert ontology.similarity_jaccard(["TO:0000005"],["TO:0000009","TO:0000002"], inherited=False) == 2/8













