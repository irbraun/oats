import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats







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




@pytest.mark.small
@pytest.mark.parametrize("input_data, expected", [
    (datasets["simple_dataset"], 10),      
])
def test_reading_in_data(input_data, expected):
	from oats.biology.dataset import Dataset
	dataset = Dataset()
	dataset.add_data(input_data)
	assert dataset.to_pandas().shape[0] == expected
	print(dataset.describe())






@pytest.mark.small
@pytest.mark.parametrize("input_data, expected, case_sensitive", [
    (datasets["simple_dataset"], 6, True),
    (datasets["simple_dataset"], 4, False),      
])
def test_collapsing_by_all_gene_names(input_data, expected, case_sensitive):
	from oats.biology.dataset import Dataset
	dataset = Dataset()
	dataset.add_data(input_data)
	dataset.collapse_by_all_gene_names(case_sensitive)
	assert dataset.to_pandas().shape[0] == expected
	print(dataset.describe())





