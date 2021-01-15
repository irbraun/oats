import pytest
import sys
sys.path.append("../oats-active")
import oats









# Removing duplicates from a list of texts and retaining its order, where duplicates don't have to be identical, they can differ by punctuation and capitalization.
@pytest.mark.parametrize("texts, expected", [
	(["Some words here.", "some words here", "Other words here.", "other WORDS here..."], ["Some words here.", "Other words here."]),
	(["a", "a", "A", "b", "c", 'B'], ["a", "b", "c"]),
	([], []),
	(["", "", "  "], [""]),
	([" ", "", "some  words"], [" ", "some  words"]),
])
def test_removal_with_cleaning(texts, expected):
	from oats.nlp.preprocess import remove_text_duplicates_retain_order
	assert remove_text_duplicates_retain_order(texts) == expected




