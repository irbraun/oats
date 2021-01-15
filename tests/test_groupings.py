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






def test_make_groupings(small_dataset):

	
	# Need a mapping between gene identifiers in this test dataset and the internal unique IDs used.
	name_to_id = small_dataset.get_name_to_id_dictionary()


	# Contents of the groupings file for the small test dataset.
	# -------------------------
	# sp1    Group1			A|B
	# sp1    Group1|Group2	A|B
	# sp2    Group1			C|D
	# sp2    Group2			C
	# sp2    Group2			D
	# sp2    Group3			C
	# sp2 	 Group4			C|D

	from oats.biology.groupings import Groupings
	path = "tests/data/small_dataset_groupings.csv"
	g = Groupings(path)
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






