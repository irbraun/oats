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
	depths = ontology.depth_dict
	assert depths["TO:0000001"] == 0
	assert depths["TO:0000002"] == 1
	assert depths["TO:0000003"] == 1
	assert depths["TO:0000004"] == 2
	assert depths["TO:0000005"] == 3
	assert depths["TO:0000006"] == 3
	assert depths["TO:0000007"] == 1
	assert depths["TO:0000008"] == 2
	assert depths["TO:0000009"] == 3



@pytest.mark.fast
def test_ontology_term_graph_based_information_content(ontology):
	"""Is the information content calculated from the graph structure what is expected?
	"""
	ic = ontology.graph_based_ic_dict
	assert ic["TO:0000001"] == 0.000
	assert ic["TO:0000002"] == 0.3690702464285426
	assert ic["TO:0000003"] == 0.3690702464285426
	assert ic["TO:0000004"] == 1.000
	assert ic["TO:0000005"] == 3.000
	assert ic["TO:0000006"] == 3.000
	assert ic["TO:0000007"] == 0.500
	assert ic["TO:0000008"] == 1.3690702464285427
	assert ic["TO:0000009"] == 3.000




@pytest.mark.fast
def test_ontology_term_inheritance(ontology):
	"""Is the number of inherited terms of each term in the graph as expected?
	"""
	inherited = ontology.subclass_dict
	assert len(inherited["TO:0000001"]) == 0
	assert len(inherited["TO:0000002"]) == 1
	assert len(inherited["TO:0000003"]) == 1
	assert len(inherited["TO:0000004"]) == 3
	assert len(inherited["TO:0000005"]) == 4
	assert len(inherited["TO:0000006"]) == 4
	assert len(inherited["TO:0000007"]) == 1
	assert len(inherited["TO:0000008"]) == 2
	assert len(inherited["TO:0000009"]) == 3





@pytest.mark.fast
def test_ontology_ic_similarity(ontology):
	"""Is the information content of the most informative common ancestor term as expected for these lists of terms?
	"""
	assert ontology.info_content_similarity(["TO:0000001"],["TO:0000002"], inherited=False) == 0
	assert ontology.info_content_similarity(["TO:0000001"],["TO:0000003"], inherited=False) == 0
	assert ontology.info_content_similarity(["TO:0000002"],["TO:0000003"], inherited=False) == 0
	assert ontology.info_content_similarity(["TO:0000003"],["TO:0000005"], inherited=False) == 0.3690702464285426
	assert ontology.info_content_similarity(["TO:0000007"],["TO:0000008"], inherited=False) == 0.5
	assert ontology.info_content_similarity(["TO:0000005"],["TO:0000009"], inherited=False) == 0

	assert ontology.info_content_similarity(["TO:0000001"],["TO:0000002","TO:0000001"], inherited=False) == 0
	assert ontology.info_content_similarity(["TO:0000003"],["TO:0000001","TO:0000009"], inherited=False) == 0
	assert ontology.info_content_similarity(["TO:0000002"],["TO:0000003","TO:0000002"], inherited=False) == 0.3690702464285426
	assert ontology.info_content_similarity(["TO:0000003"],["TO:0000005","TO:0000002"], inherited=False) == 0.3690702464285426
	assert ontology.info_content_similarity(["TO:0000008"],["TO:0000008","TO:0000007"], inherited=False) == 1.3690702464285427
	assert ontology.info_content_similarity(["TO:0000005"],["TO:0000009","TO:0000002"], inherited=False) == 0.3690702464285426


@pytest.mark.fast
def test_ontology_jaccard_similarity(ontology):
	"""Is the Jaccard similarity between the given lists of terms as expected?
	"""
	assert ontology.jaccard_similarity(["TO:0000001"],["TO:0000002"], inherited=False) == 1/2
	assert ontology.jaccard_similarity(["TO:0000001"],["TO:0000003"], inherited=False) == 1/2
	assert ontology.jaccard_similarity(["TO:0000002"],["TO:0000003"], inherited=False) == 1/3
	assert ontology.jaccard_similarity(["TO:0000003"],["TO:0000005"], inherited=False) == 2/5
	assert ontology.jaccard_similarity(["TO:0000007"],["TO:0000008"], inherited=False) == 2/3
	assert ontology.jaccard_similarity(["TO:0000005"],["TO:0000009"], inherited=False) == 1/8

	assert ontology.jaccard_similarity(["TO:0000001"],["TO:0000002","TO:0000001"], inherited=False) == 1/2
	assert ontology.jaccard_similarity(["TO:0000003"],["TO:0000001","TO:0000009"], inherited=False) == 1/5
	assert ontology.jaccard_similarity(["TO:0000002"],["TO:0000003","TO:0000002"], inherited=False) == 2/3 
	assert ontology.jaccard_similarity(["TO:0000003"],["TO:0000005","TO:0000002"], inherited=False) == 2/5
	assert ontology.jaccard_similarity(["TO:0000008"],["TO:0000008","TO:0000007"], inherited=False) == 3/3
	assert ontology.jaccard_similarity(["TO:0000005"],["TO:0000009","TO:0000002"], inherited=False) == 2/8













