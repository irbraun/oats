import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats






@pytest.fixture
def small_dataset():
	with open("tests/data/small_dataset.csv") as f:
		df = pd.read_csv(f)
		from oats.biology.dataset import Dataset
		dataset = Dataset()
		dataset.add_data(df)
		return(dataset)









@pytest.mark.small
def test_make_groupings(small_dataset):

	
	# Need a mapping between 'gene names' of this test dataset and the internal unique IDs used.
	small_dataset.collapse_by_all_gene_names()
	name_to_id = small_dataset.get_name_to_id_dictionary()


	# For the species sp1 file:
	# -------------------------
	# Group1			A|B
	# Group1|Group2		A|B
	#
	#
	#
	# For the species sp2 file:
	# -------------------------
	# Group1			C|D
	# Group2			C
	# Group2			D
	# Group3			C
	# Group4			C|D

	from oats.biology.groupings import Groupings
	d = {"sp1":"tests/data/small_dataset_groupings_sp1.csv", "sp2":"tests/data/small_dataset_groupings_sp2.csv"}
	g = Groupings(d, source="csv", name_mapping={})
	d1,d2 = g.get_groupings_for_dataset(small_dataset)
	id_to_groups = d1
	group_to_ids = d2

	# Do the genes present in the dataset map to the right groups?
	assert set(id_to_groups[name_to_id["A"]]) == set(["Group1", "Group2"])
	assert set(id_to_groups[name_to_id["B"]]) == set(["Group1", "Group2"])
	assert set(id_to_groups[name_to_id["C"]]) == set(["Group1", "Group2", "Group3", "Group4"])
	assert set(id_to_groups[name_to_id["D"]]) == set(["Group1", "Group2", "Group4"])
	# Do the lists that are those values not contain any duplicates?
	assert len(id_to_groups[name_to_id["A"]]) == 2
	assert len(id_to_groups[name_to_id["B"]]) == 2
	assert len(id_to_groups[name_to_id["C"]]) == 4
	assert len(id_to_groups[name_to_id["D"]]) == 3
	# Do the groups present in the dataset map to the right genes?
	assert set(group_to_ids["Group1"]) == set([name_to_id["A"], name_to_id["B"], name_to_id["C"], name_to_id["D"]])
	assert set(group_to_ids["Group2"]) == set([name_to_id["A"], name_to_id["B"], name_to_id["C"], name_to_id["D"]])
	assert set(group_to_ids["Group3"]) == set([name_to_id["C"]])
	assert set(group_to_ids["Group4"]) == set([name_to_id["C"], name_to_id["D"]])
	# Do the lists that are those values not contain any duplicates?
	assert len(group_to_ids["Group1"]) == 4
	assert len(group_to_ids["Group2"]) == 4
	assert len(group_to_ids["Group3"]) == 1
	assert len(group_to_ids["Group4"]) == 2












