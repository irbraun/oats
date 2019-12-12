import pytest
import sys
sys.path.append("../oats")
import oats









# Searching for an exact pattern in a larger string.
@pytest.mark.parametrize("text, pattern, q, expected", [
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seeds", 235235, True),                # Matching an entire word.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seed", 235235, True),                 # Matching part of a word.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "mature", 235235, False),              # Not matching a word if wrong case used.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "hada", 235235, False),                # Not matching two words without space between.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "had a", 235235, True),                # Matching two words if the space is included.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "wrinkled phenotype", 235235, False),  # Not matching if punctuation is missing.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "", 235235, False),                    # An emtpy string is considered to not be found.
])
def test_binary_search_rabin_karp(text, pattern, q, expected):
    from oats.nlp.search import binary_search_rabin_karp
    assert binary_search_rabin_karp(pattern, text, q) == expected



# Checking for any exact occurence of any pattern in a list in a larger string.
@pytest.mark.parametrize("text, patterns, q, expected", [
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["seeds", "from", "a"], 235235, True),             # All the patterns can be found.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["seeds", "hsodf", "ksopw"], 235235, True),        # One of the patterns can be found, the others don't match.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDD-treated", "had", "had"], 235235, True),     # One of the matching patterns is listed twice.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["-treated", "hsodf"], 235235, True),              # A partial-word pattern matches.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDDtreated", "hsodf"], 235235, False),          # One of the patterns would match but missing punctuation.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDD-treated", "had", "hsodf"], 235235, True),   # Some of the patterns can be found, one can't.
])
def test_search_for_any_rabin_karp(text, patterns, q, expected):
    from oats.nlp.search import search_for_any_rabin_karp
    assert search_for_any_rabin_karp(patterns, text, q) == expected



# Checking for which patterns exactly occur in larger string.
@pytest.mark.parametrize("text, patterns, q, expected", [
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["seeds","from","a"], 235235, ["seeds","from","a"]),                   # All the patterns can be found.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["seeds","hsodf","ksopw"], 235235, ["seeds"]),                         # One of the patterns can be found, the others don't match.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDD-treated","had","had"], 235235, ["TCDD-treated","had","had"]),   # One of the matching patterns is listed twice, duplicate is not removed.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["-treated","hsodf"], 235235, ["-treated"]),                           # A partial-word pattern matches.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDDtreated","hsodf"], 235235, []),                                  # One of the patterns would match but missing punctuation. Empty list return when nothing matches.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", ["TCDD-treated","had","hsodf"], 235235, ["TCDD-treated","had"]),       # Two of the patterns can be found, one can't.
    ("Mature seeds from TCDD-treated plants had a 'wrinkled' phenotype.", [], 235235, []),                                                       # The empty list case is handled, returned as no patterns found.
])
def test_search_for_all_rabin_karp(patterns, text, q, expected):
    from oats.nlp.search import search_for_all_rabin_karp
    assert search_for_all_rabin_karp(patterns, text, q) == expected












# Searching for a fuzzy match to a pattern in a larger string, looking for either local or global alignments.
@pytest.mark.parametrize("text, pattern, threshold, local, expected", [
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seeds", 1.00, 1, True),                   # A perfect local match is found when the threshold is 1.0.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seeds", 0.90, 1, True),                   # A perfect local match is still found when threshold is decreased.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "s_eds", 1.00, 1, False),                  # A partial local match is not found when the threshold is 1.0.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "s_eds", 0.75, 1, True),                   # A partial local macch is found when the threshold is decreased.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seed", 1.00, 1, True),                    # Substrings are considered perfect local matches.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seeds from", 1.00, 1, True),              # Perfect local matches can be found across word boundaries.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", "seedsfrom", 0.80, 1, True),               # Partial local matches can be found across word boundaries.
    ("Mature seeds ... 'wrinkled' phenotype.", "Mature seeds ... 'wrinkled' phenotype.", 1.00, 0, True),                            # A perfect global match is found when the threshold is 1.0.
    ("Mature seeds ... 'wrinkled' phenotype.", "Mature seeds ... 'wrinkled' phenotype.", 0.90, 0, True),                            # A perfect global match is still found when the threshold is decreased.
    ("Mature seeds ... 'wrinkled' phenotype.", "Mature sxxds ... 'wrinkled' phenotype.", 1.00, 0, False),                           # A partial global match is not found when the threshold is 1.0.
    ("Mature seeds ... 'wrinkled' phenotype.", "Mature sxxds ... 'wrinkled' phenotype.", 0.90, 0, True),                            # A partial global match is found when the threshold is decreased.
    ("Mature seeds ... 'wrinkled' phenotype.", "seeds", 1.00, 0, False),                                                            # Local alignments are not found when searching for global one.

])
def test_binary_search_karp(text, pattern, threshold, local, expected):
    from oats.nlp.search import binary_search_fuzzy
    assert binary_search_fuzzy(pattern, text, threshold, local) == expected





# Checking for a fuzzy match of any pattern in a list in a larger string, using local alignments.
@pytest.mark.parametrize("text, patterns, threshold, local, expected", [
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seeds"], 1.00, 1, True),                 # A perfect local match is found when only one pattern is passed in.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seeds","s_eds"], 1.00, 1, True),         # Case of strict threhsold and atleast one perfect match.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seed_","s_eds"], 1.00, 1, False),        # Case of strict threshold and no perfect matches, just partial ones.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seed_","s_eds"], 0.75, 1, True),         # Case of some partial matches but a lowered threshold.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["",""], 0.00, 1, True),                   # Empty strings are considered matches if the threshold is at at the minimum only.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["",""], 0.01, 1, False),                  # Empty strings are considered matches if the threshold is at at the minimum only.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", [], 0.00, 1, False),                       # An empty list of patterns will never find matches.                 
])
def test_search_for_any_fuzzy(text, patterns, threshold, local, expected):
    from oats.nlp.search import search_for_any_fuzzy
    assert search_for_any_fuzzy(patterns, text, threshold, local) == expected





# Checking for which fuzzy matches of any pattern in a list in a larger string, using local alignments.
@pytest.mark.parametrize("text, patterns, threshold, local, expected", [
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seeds"], 1.00, 1, ["seeds"]),                    # A perfect local match is found when only one pattern is passed in.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seeds","s_eds"], 1.00, 1, ["seeds"]),            # When the threshold is 1, only the perfect local match is returned.
    ("Mature seeds from TCDD-treated plants had a characteristic 'wrinkled' phenotype.", ["seeds","s_eds"], 0.75, 1, ["seeds","s_eds"]),    # When the threshold is decreased, both the perfect and partial match are returned.
])
def test_search_for_any_fuzzy(text, patterns, threshold, local, expected):
    from oats.nlp.search import search_for_all_fuzzy
    assert search_for_all_fuzzy(patterns, text, threshold, local) == expected





















