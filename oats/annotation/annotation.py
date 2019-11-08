from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import os
import sys
import glob
import math
import re

from oats.nlp.search import binary_search_rabin_karp











def annotate_using_rabin_karp(object_dict, ontology, fixcase=1):
	"""Build a dictionary of annotations using the Rabin Karp algorithm.

	Args:
	    object_dict (dict): Mapping from IDs to natural language descriptions.
	    ontology (Ontology): Ontology object with specified terms.
	    fixcase (int, optional): Set to 1 to make words from ontologies and 
	    the searched text both lowercase, set to 0 else.

	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.
	"""
	annotations = defaultdict(list)
	prime = 193
	for identifer,description in object_dict.items():
		annotations[identifer].extend([])
		for word,term_list in ontology.reverse_term_dict.items():
			if fixcase==1:
				word = word.lower()
				description = description.lower()
			if binary_search_rabin_karp(word, description, prime):
				annotations[identifer].extend(term_list)
	return(annotations)










def annotate_using_noble_coder(object_dict, path_to_jarfile, *ontology_names, precise=1):
	"""Build a dictionary of annotations using NOBLE Coder.

	Args:
	    object_dict (dict): Mapping from object IDs to natural language descriptions.
	    path_to_jarfile (str): Path to the jar file for the NOBLE Coder tool.
	    ontology_names (list): Strings used to find the correct terminology file, should match obo files too.
	    precise (int, optional): Set to 1 to do precise matching, set to 0 to accept partial matches.

	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.

	Raises:
	    FileNotFoundError: NOBLE Coder can't find the terminology file matching this ontology.
	"""



	# Configuration for running the NOBLE Coder script.
	tempfiles_directory = "temp_textfiles"
	output_directory = "temp_output"
	if not os.path.exists(tempfiles_directory):
		os.makedirs(tempfiles_directory)
	default_results_filename = "RESULTS.tsv"
	default_results_path = os.path.join(output_directory,default_results_filename)
	if precise == 1:
		specificity = "precise-match"
	else:
		specificity = "partial-match"


	# Generate temporary text files for each of the text descriptions.
	# Identifiers for descriptions are encoded into the filenames themselves.
	annotations = {identifier:[] for identifier in object_dict.keys()}
	for identifier,description in object_dict.items():
		tempfile_path = os.path.join(tempfiles_directory, f"{identifier}.txt")
		with open(tempfile_path, "w") as file:
			file.write(description)

	# Use all specified ontologies to annotate each text file. 
	# Also NOBLE Coder will check for a terminology file matching this ontology, make sure it's there.
	for ontology_name in ontology_names:
		expected_terminology_file = os.path.expanduser(os.path.join("~",".noble", "terminologies", f"{ontology_name}.term"))
		if not os.path.exists(expected_terminology_file):
			raise FileNotFoundError(expected_terminology_file)
		os.system(f"java -jar {path_to_jarfile} -terminology {ontology_name} -input {tempfiles_directory} -output {output_directory} -search '{specificity}' -score.concepts")
		default_results_filename = "RESULTS.tsv"		
		for identifier,term_list in _parse_noble_coder_results(default_results_path).items():
			# Need to convert identifier back to an integer because it's being read from a file name.
			# NOBLE Coder finds every occurance of a matching, reduce this to form a set.
			identifier = int(identifier)
			term_list = list(set(term_list))
			annotations[identifier].extend(term_list)



	# Cleanup and return the annotation dictionary.
	_cleanup_noble_coder_results(output_directory, tempfiles_directory)
	return(annotations)










def _parse_noble_coder_results(results_filename):
	"""
	Returns a mapping from object IDs to ontology term IDs inferred from reading
	a NOBLE Coder output file, supports the above method.

	Args:
	    results_filename (str): Path to the output file created by NOBLE Coder.

	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.
	"""
	df = pd.read_csv(results_filename, usecols=["Document", "Matched Term", "Code"], sep="\t")
	annotations = defaultdict(list)
	for row in df.itertuples():
		textfile_processed = row[1]
		identifer = str(textfile_processed.split(".")[0])
		tokens_matched = row[2].split()
		ontology_term_id = row[3]
		annotations[identifer].append(ontology_term_id)
	return(annotations)








def _cleanup_noble_coder_results(output_directory, textfiles_directory):
	"""
	Removes all directories and files created and used by running NOBLE Coder.

	Args:
	    output_directory (str): Path to directory containing NOBLE Coder outputs.
	    textfiles_directory (str): Path to the directory of input text files.
	"""

	# Expected paths to each object that should be removed.
	html_file = os.path.join(output_directory,"index.html")
	results_file = os.path.join(output_directory,"RESULTS.tsv")
	properties_file = os.path.join(output_directory,"search.properties")
	reports_directory = os.path.join(output_directory,"reports")

	# Safely remove everything in the output directory.
	if os.path.isfile(html_file):
		os.remove(html_file)
	if os.path.isfile(results_file):
		os.remove(results_file)
	if os.path.isfile(properties_file):
		os.remove(properties_file)
	for filepath in glob.iglob(os.path.join(reports_directory,"*.html")):
		os.remove(filepath)
	os.rmdir(reports_directory)
	os.rmdir(output_directory)

	# Safely remove everything in the text file directory.
	for filepath in glob.iglob(os.path.join(textfiles_directory,"*.txt")):
		os.remove(filepath)
	os.rmdir(textfiles_directory)








def write_annotations_to_tsv_file(annotations_dict, annotations_output_path):
	"""Create a tsv file of annotations that is compatable with fastsemsim.
	
	Args:
	    annotations_dict (dict): Mapping from IDs to lists of ontology term IDs.
	    annotations_output_file (str): Path to the output file to create.
	"""
	outfile = open(annotations_output_path,"w")
	for identifer,term_list in annotations_dict.items():
		row_values = [str(identifer)]
		row_values.extend(term_list)
		outfile.write("\t".join(row_values).strip()+"\n")
	outfile.close()







def read_annotations_from_tsv_file(annotations_input_path):
	"""Get a dictionary mapping ID's to list of ontology term ID's.

	Args:
	    annotations_input_file (str): Path to the input file to open.

	Returns:
	    dict: Mapping from object IDs to lists of ontology term IDs.
	"""
	infile = open(annotations_input_path, "r")
	annotations_dict = {}
	separator = "\t"
	for line in infile.read():
		row_values = line.strip().split(separator)
		identifier = row_values[0]
		term_ids = row_values[1:len(row_values)]
		annotations_dict[identifer] = term_ids
	return(annotations_dict)










