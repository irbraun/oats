import pytest
import sys
import pandas as pd


sys.path.append("../oats")
from oats.biology.dataset import Dataset




# This is what the simple dataset looks like when read in.

# species	unique_gene_identifiers	other_gene_identifiers	gene_models	descriptions	annotations	sources
# sp1	A	Z	A	sentence from line 1	XO:0000111	Y
# sp1	a	Z	a	sentence from line 2	XO:0000111	Y
# sp1	B	Z	B	sentence from line 3	XO:0000111	Y
# sp1	B	Z	B	sentence from line 4	XO:0000111	Y
# sp2	C	Z		sentence from line 5	XO:0000111	Y
# sp2	C	Z		sentence from line 6	XO:0000111	Y
# sp2	C	Z		sentence from line 7	XO:0000111	Y
# sp2	C	Z		sentence from line 8	XO:0000111	Y
# sp2	d	Z		sentence from line 9	XO:0000111	Y
# sp2	D	Z		sentence from line 10	XO:0000111	Y





datasets = {
	"simple_dataset":pd.read_csv("/Users/irbraun/oats/tests/data/small_dataset.csv"),
}






@pytest.mark.parametrize("input_data, expected", [
    (datasets["simple_dataset"], 10),      
])
def test_reading_in_data(input_data, expected):
	
	
	# Using the constructor method and retaining the original IDs (and therefore size) of the dataset.
	dataset = Dataset(data=input_data, keep_ids=True)
	assert dataset.to_pandas().shape[0] == expected






@pytest.mark.parametrize("input_data, expected, case_sensitive", [
    (datasets["simple_dataset"], 6, True),
    (datasets["simple_dataset"], 4, False),      
])
def test_collapsing_by_all_gene_names(input_data, expected, case_sensitive):
	

	# Using the constructor and retaining the original size, then collapsing by calling the hidden method.
	# This is not the intended way to do this but it should always work.
	dataset = Dataset(data=input_data, keep_ids=True)
	dataset._collapse_by_all_gene_names(case_sensitive)
	assert dataset.to_pandas().shape[0] == expected


	# Create a blank dataset then add this information to it, and it should be automatically collapsed.
	# This is one of the intended ways to do this.
	dataset = Dataset()
	dataset.add_data(new_data=input_data, case_sensitive=case_sensitive)
	assert dataset.to_pandas().shape[0] == expected


	# Use the constructor and don't specify that IDs should be kept, so it gets automatically collapsed.
	# This is one of the intended ways to do this.
	dataset = Dataset(data=input_data, keep_ids=False, case_sensitive=case_sensitive)
	assert dataset.to_pandas().shape[0] == expected






def test_json():
	input_data = datasets["simple_dataset"]
	dataset = Dataset(data=input_data, keep_ids=True)
	import json
	with open("/Users/irbraun/Desktop/testing.json", "w") as f:
		json.dump(dataset.to_json(), f, indent=4)




