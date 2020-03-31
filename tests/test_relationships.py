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
def test_read_relationships(small_dataset):

	
	# Need a mapping between 'gene names' of this test dataset and the internal unique IDs used.
	small_dataset.collapse_by_all_gene_names()
	name_to_id = small_dataset.get_name_to_id_dictionary()

	# From the txt file specifying relationships between genes.
	# A		B	0.234
	# A		C	0.645
	# C		B	0.123
	# D		E	0.111
	# D		F	0.975

	from oats.biology.relationships import AnyInteractions
	path = "tests/data/small_dataset_relationships.txt"
	interactions = AnyInteractions(name_to_id, path)
	df = interactions.get_df()
	ids = interactions.get_ids()

	# The IDs that correspond to A, B, and C should be in the IDs that were mentioned in the file.
	assert set(ids) == set([name_to_id["A"], name_to_id["B"], name_to_id["C"], name_to_id["D"]])
	assert len(ids) == 4

	# The dataframe returned should contain six rows because only three involve pairs of genes where both 
	# are in the dataset that is being used (A, B, and C), and then the reverse of each relationship is 
	# included too because that is the default behavior that makes it easier to merge this dataframe with
	# other existing dataframes.
	assert df.shape == (6,3)











