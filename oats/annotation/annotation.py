from collections import defaultdict
import pandas as pd
import os
import sys
import glob

from oats.nlp.search import binary_robinkarp_match, binary_fuzzy_match











def annotate_using_rabin_karp(ids_to_texts, ontology, fixcase=1):
	"""
	Build a dictionary of annotations using the Rabin Karp algorithm. This is useful for finding
	instances of ontology terms in 


	Args:
		ids_to_texts (dict of int:str): Mapping from unique integer IDs to natural language text strings.

		ontology (oats.annotation.Ontology): Object of the ontology to be used. 

		fixcase (int, optional): Set to 1 to normalize all strings before matching, set to 0 to ignore this option.

	Returns:
		dict of int:list of str: Mapping from unique integer IDs to lists of ontology term IDs.
	"""
	annotations = defaultdict(list)
	prime = 193
	for identifer,description in ids_to_texts.items():
		annotations[identifer].extend([])
		for word,term_list in ontology.token_to_terms.items():
			if fixcase==1:
				word = word.lower()
				description = description.lower()
			if binary_robinkarp_match(word, description, prime):
				annotations[identifer].extend(term_list)
	return(annotations)









def annotate_using_fuzzy_matching(ids_to_texts, ontology, threshold=0.90, fixcase=1, local=1):
	"""Build a dictionary of annotations using fuzzy string matching.
	
	Args:
		ids_to_texts (dict of int:str): Mapping from unique integer IDs to natural language text strings.
		
		ontology (oats.annotation.Ontology): Ontology object with specified terms.
		
		threshold (float, optional): Value ranging from 0 to 1, the similarity threshold for string matches.
		
		fixcase (int, optional): Set to 1 to normalize all strings before matching, set to 0 to ignore this option.
		
		local (int, optional): Set the alignment method, 0 for global and 1 for local. Local alignment should 
		always be used for annotating ontology terms to long strings of text.
	
	Returns:
		dict of int:list of str: Mapping from unique integer IDs to lists of ontology term IDs.
	"""
	annotations = defaultdict(list)
	for identifier,description in ids_to_texts.items():
		annotations[identifier].extend([])
		for word, term_list in ontology.token_to_terms.items():
			if fixcase==1:
				word = word.lower()
				description = description.lower()
			if binary_fuzzy_match(word, description, threshold, local):
				annotations[identifier].extend(term_list)
	return(annotations)









def annotate_using_noble_coder(ids_to_texts, jar_path, *ontology_names, precise=1):
	"""Build a dictionary of annotations using NOBLE Coder (Tseytlin et al., 2016).

	Args:
		ids_to_texts (dict of int:str): Mapping from unique integer IDs to natural language text strings.
		
		jar_path (str): Path of the NOBLE Coder jar file.

		ontology_names (list of str): Names of the ontologies (e.g., "pato", "po") used to find matching NOBLE Coder 
		terminology files (e.g., pato.term, po.term) in ~/.noble/terminologies.

		precise (int, optional): Set to 1 to do precise matching, set to 0 to accept partial matches.

	Returns:
		dict of int:list of str: Mapping from unique integer IDs to lists of ontology term IDs.

	Raises:
		FileNotFoundError: NOBLE Coder cannot find the terminology file matching this ontology.
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
	annotations = {identifier:[] for identifier in ids_to_texts.keys()}
	for identifier,description in ids_to_texts.items():
		tempfile_path = os.path.join(tempfiles_directory, f"{identifier}.txt")
		with open(tempfile_path, "w") as file:
			file.write(description)

	# Use all specified ontologies to annotate each text file. 
	# Also NOBLE Coder will check for a terminology file matching this ontology, make sure it's there.
	for ontology_name in ontology_names:
		expected_terminology_file = os.path.expanduser(os.path.join("~",".noble", "terminologies", f"{ontology_name}.term"))
		if not os.path.exists(expected_terminology_file):
			raise FileNotFoundError(expected_terminology_file)
		os.system(f"java -jar {jar_path} -terminology {ontology_name} -input {tempfiles_directory} -output {output_directory} -search '{specificity}' -score.concepts")
		default_results_filename = "RESULTS.tsv"		
		for identifier,term_list in _parse_noble_coder_results(default_results_path).items():
			# Need to convert identifier back to an integer because it's being read from a file name.
			# NOBLE Coder finds every occurance of a matching, reduce this to form a set.
			identifier = int(identifier)
			term_list = list(set(term_list))
			term_list = [term_id.replace("_",":") for term_id in term_list]
			annotations[identifier].extend(term_list)



	# Cleanup and return the annotation dictionary.
	_cleanup_noble_coder_results(output_directory, tempfiles_directory)
	return(annotations)












def _parse_noble_coder_results(results_filename):
	"""
	Translates the generated NOBLE Coder output file into a dictionary of annotations.

	Args:
		results_filename (str): Path of the output file created by NOBLE Coder.

	Returns:
		dict of int:list of str: Mapping from unique integer IDs to lists of ontology term IDs.
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
		output_directory (str): Path of the directory containing the NOBLE Coder outputs.

		textfiles_directory (str): Path of the directory of input text files.
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

















def write_annotations_to_file(annotations_dict, annotations_output_path, sep="\t"):
	""" Write a dictionary of annotations to a file.
	
	Args:
		annotations_dict (dict of int:list of str): Mapping from unique integer IDs to lists of ontology term IDs.

		annotations_output_file (str): Path of the output file that will be created.
	"""
	outfile = open(annotations_output_path,"w")
	for identifer,term_list in annotations_dict.items():
		row_values = [str(identifer)]
		row_values.extend(term_list)
		outfile.write(sep.join(row_values).strip()+"\n")
	outfile.close()




def read_annotations_from_file(annotations_input_path, sep="\t"):
	""" Read a file of annotations and produce a dictionary. 

	Args:
		annotations_input_file (str): Path of the input annotations file to read.

	Returns:
		dict of int:list of str: Mapping from unique integer IDs to lists of ontology term IDs.
	"""
	infile = open(annotations_input_path, "r")
	annotations_dict = {}
	for line in infile.read():
		row_values = line.strip().split(sep)
		identifier = row_values[0]
		term_ids = row_values[1:len(row_values)]
		annotations_dict[identifer] = term_ids
	return(annotations_dict)










