import pytest
import sys
sys.path.append("../oats")
import oats









# Searching for an exact pattern in a larger string.
@pytest.mark.parametrize("pattern, text, q, expected", [
    ("seeds", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, True),
    ("seed", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, True),
    ("hada", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, False),
    ("wrinkled phenotype", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, False),
    ("DNA", "DNA methylation carried out by different methyltransferase classes.", 239587, True),
    ("dna", "DNA methylation carried out by different methyltransferase classes.", 239587, False),
    ("2,3,7,8", "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, True),
    ("2378", "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, False),
])
def test_binary_search_karp(pattern, text, q, expected):
    from oats.nlp.search import binary_search_rabin_karp
    assert binary_search_rabin_karp(pattern, text, q) == expected





# Checking for any exact occurence of any pattern in a list in a larger string.
@pytest.mark.parametrize("patterns, text, q, expected", [
    (["Mature","seeds","from"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, True),
    (["Mature","seeds","From"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, True),
    (["hada"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, False),
    (["wrinkled phenotype","'wrinkled' phenotype"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, True),
    (["DNA"], "DNA methylation carried out by different methyltransferase classes.", 239587, True),
    (["dna"], "DNA methylation carried out by different methyltransferase classes.", 239587, False),
    (["2,3,7,8"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, True),
    (["2378"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, False),
    (["2378","2,3,7,8"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, True),
])
def test_search_for_any_rabin_karp(patterns, text, q, expected):
    from oats.nlp.search import search_for_any_rabin_karp
    assert search_for_any_rabin_karp(patterns, text, q) == expected




# Checking for which patterns exactly occur in larger string.
@pytest.mark.parametrize("patterns, text, q, expected", [
    (["Mature","seeds","from"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, ["Mature","seeds","from"]),
    (["Mature","seeds","From"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, ["Mature","seeds"]),
    (["hada"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, []),
    (["wrinkled phenotype","'wrinkled' phenotype"], "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 235235, ["'wrinkled' phenotype"]),
    (["DNA"], "DNA methylation carried out by different methyltransferase classes.", 239587, ["DNA"]),
    (["dna"], "DNA methylation carried out by different methyltransferase classes.", 239587, []),
    (["2,3,7,8"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, ["2,3,7,8"]),
    (["2378"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, []),
    (["2378","2,3,7,8"], "Plants were exposed to various concentrations of the most toxic congener of dioxins, 2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD).", 235345, ["2,3,7,8"]),
])
def test_search_for_all_rabin_karp(patterns, text, q, expected):
    from oats.nlp.search import search_for_all_rabin_karp
    assert search_for_all_rabin_karp(patterns, text, q) == expected






# Searching for a fuzzy match to a pattern in a larger string.
@pytest.mark.parametrize("pattern, text, threshold, local, expected", [
    ("seeds", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.90, 1, True),
    ("seeds", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.90, 0, False),
    ("seedss", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.90, 1, False),
    ("seedss", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.80, 1, True),
    ("seed3z", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.90, 1, False),
    ("seed3z", "Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype, due to a two-thirds reduction in storage oil content.", 0.60, 1, True),
    ("Mature seeds ... oil", "Mature seeds ... oil content.", 0.90, 1, True),
    ("Mature seeds ... oil", "Mature seeds ... oil content.", 0.90, 0, False),
    ("Mature seeds ... oil", "Mature seeds ... oil content.", 0.70, 0, True),
])
def test_binary_search_karp(pattern, text, threshold, local, expected):
    from oats.nlp.search import binary_search_fuzzy
    assert binary_search_fuzzy(pattern, text, threshold, local) == expected






