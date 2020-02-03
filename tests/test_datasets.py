import pytest
import sys
import pandas as pd
sys.path.append("../oats")
import oats







# species	gene_names	description	term_ids
# ath	A	sentence from line 1	XO:0000111
# ath	A	sentence from line 2	XO:0000111
# ath	B	sentence from line 3	XO:0000111
# ath	B	sentence from line 4	XO:0000111
# zma	C	sentence from line 5	XO:0000111
# zma	C	sentence from line 6	XO:0000111
# zma	C	sentence from line 7	XO:0000111
# zma	C	sentence from line 8	XO:0000111
# zma	D	sentence from line 9	XO:0000111
# zma	D	sentence from line 10	XO:0000111




datasets = {
	"simple_dataset_1":pd.read_csv("/Users/irbraun/oats/tests/test_data/small_dataset.csv"),
	"simple_dataset_2":pd.read_csv("/Users/irbraun/oats/tests/test_data/small_dataset.csv"),
	"simple_dataset_3":pd.read_csv("/Users/irbraun/oats/tests/test_data/small_dataset.csv"),
	"simple_dataset_4":pd.read_csv("/Users/irbraun/oats/tests/test_data/small_dataset.csv"),
}






@pytest.mark.parametrize("input_data, expected", [
    (datasets["simple_dataset_1"], 10),   
    (datasets["simple_dataset_2"], 10),   
    (datasets["simple_dataset_3"], 10),    
    (datasets["simple_dataset_4"], 10),        
])
def test_reading_in_data(input_data, expected):
	from oats.datasets.dataset import Dataset
	dataset = Dataset()
	dataset.add_data(input_data)
	assert dataset.to_pandas().shape[0] == expected







@pytest.mark.parametrize("input_data, expected", [
    (datasets["simple_dataset_1"], 4),   
    (datasets["simple_dataset_2"], 4),   
    (datasets["simple_dataset_3"], 4),    
    (datasets["simple_dataset_4"], 4),        
])
def test_collapsing_by_all_gene_names(input_data, expected):
	from oats.datasets.dataset import Dataset
	dataset = Dataset()
	dataset.add_data(input_data)
	dataset.collapse_by_all_gene_names()
	assert dataset.to_pandas().shape[0] == expected






