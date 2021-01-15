import pytest
import sys
import pandas as pd
sys.path.append("../oats-active")
import oats









@pytest.fixture
def po():
	from oats.annotation.ontology import Ontology
	ontology_filename = "../phenologs-with-oats/ontologies/po.obo"   
	ontology = Ontology(ontology_filename)
	return(ontology)


@pytest.fixture
def pato():
	from oats.annotation.ontology import Ontology
	ontology_filename = "../phenologs-with-oats/ontologies/pato.obo"   
	ontology = Ontology(ontology_filename)
	return(ontology)






# Sampled term IDs and their commented labels from each of these two real ontologies.
pato_term_id_1 = "PATO:0000119" # height
pato_term_id_2 = "PATO:0000569" # decreased height
po_term_id_1 = "PO:0000036"		# leaf vascular system
po_term_id_2 = "PO:0000059"		# root initial cell












def test_inherited_method_behavior(pato, po):
	"""Are the depths of each term what are expected?
	"""

	# Using an ontology where the term ID is not recognized should just return the term itself.
	# In other words, this method is intended for now to fail silently to retrieve any inherited
	# terms of this given term from the ontology being used.
	assert po.inherited(pato_term_id_1) == [pato_term_id_1]
	assert po.inherited(pato_term_id_2) == [pato_term_id_2]
	assert pato.inherited(po_term_id_1) == [po_term_id_1]
	assert pato.inherited(po_term_id_2) == [po_term_id_2]
	
	# However, if the ontology being used is the one that the given term is actually located as 
	# as a node in, then the list of inherited terms should both include the passed in term ID and
	# included some other ones that are superclasses above it in the ontology graph, as long as the
	# passed in terms are not roots, which in this case they are not.

	assert len(po.inherited(po_term_id_1)) > 1
	assert len(po.inherited(po_term_id_1)) > 1
	assert len(pato.inherited(pato_term_id_1)) > 1
	assert len(pato.inherited(pato_term_id_1)) > 1

	assert po_term_id_1 in po.inherited(po_term_id_1)
	assert po_term_id_2 in po.inherited(po_term_id_2)
	assert pato_term_id_1 in pato.inherited(pato_term_id_1)
	assert pato_term_id_2 in pato.inherited(pato_term_id_2)
























