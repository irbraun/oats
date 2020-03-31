import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats







# species	gene_names   gene_synonyms	description	term_ids
# ath	A	Z   sentence from line 1	XO:0000111
# ath	A	Z   sentence from line 2	XO:0000111
# ath	B	Z   sentence from line 3	XO:0000111
# ath	B	Z   sentence from line 4	XO:0000111
# zma	C	Z   sentence from line 5	XO:0000111
# zma	C	Z   sentence from line 6	XO:0000111
# zma	C	Z   sentence from line 7	XO:0000111
# zma	C	Z   sentence from line 8	XO:0000111
# zma	D	Z   sentence from line 9	XO:0000111
# zma	D	Z   sentence from line 10	XO:0000111




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
@pytest.mark.parametrize("input_data, expected", [
    (datasets["simple_dataset"], 4),       
])
def test_collapsing_by_all_gene_names(input_data, expected):
	from oats.biology.dataset import Dataset
	dataset = Dataset()
	dataset.add_data(input_data)
	dataset.collapse_by_all_gene_names()
	assert dataset.to_pandas().shape[0] == expected
	print(dataset.describe())





