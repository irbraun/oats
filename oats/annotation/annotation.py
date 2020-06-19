from collections import defaultdict
from scipy.stats import fisher_exact
import pandas as pd
import os
import sys
import glob


from oats.nlp.search import binary_robinkarp_match, binary_fuzzy_match
from oats.utils.utils import flatten











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









def annotate_using_noble_coder(ids_to_texts, jar_path, ontology_name, precise=1, output=None):
	"""Build a dictionary of annotations using NOBLE Coder (Tseytlin et al., 2016).
	
	Args:
		ids_to_texts (dict of int:str): Mapping from unique integer IDs to natural language text strings.
	
		jar_path (str): Path of the NOBLE Coder jar file.
	
		ontology_name (str): Name of the ontology (e.g., "pato", "po") used to find matching a NOBLE Coder terminology file (e.g., pato.term, po.term) in ~/.noble/terminologies.
	
		precise (int, optional): Set to 1 to do precise matching, set to 0 to accept partial matches.
	
		output (str, optional): Path to a text file where the stdout from running NOBLE Coder should be redirected. If not provided, this output is redirected to a temporary file and deleted. 
	
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

	if not os.path.exists(output_directory):
		os.makedirs(output_directory)





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
	expected_terminology_file = os.path.expanduser(os.path.join("~",".noble", "terminologies", f"{ontology_name}.term"))
	if not os.path.exists(expected_terminology_file):
		raise FileNotFoundError(expected_terminology_file)

	if output is not None:
		stdout_path = output
	else:
		stdout_path = os.path.join(output_directory,"nc_stdout.txt")

	os.system(f"java -jar {jar_path} -terminology {ontology_name} -input {tempfiles_directory} -output {output_directory} -search '{specificity}' -score.concepts > {stdout_path}")	
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
	stdout_file = os.path.join(output_directory,"nc_stdout.txt")
	html_file = os.path.join(output_directory,"index.html")
	results_file = os.path.join(output_directory,"RESULTS.tsv")
	properties_file = os.path.join(output_directory,"search.properties")
	reports_directory = os.path.join(output_directory,"reports")

	# Safely remove everything in the output directory.
	if os.path.isfile(stdout_file):
		os.remove(stdout_file)
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












def _get_term_name(i, ontology):
    """ Small helper function for the function below.
    """
    try: 
    	return(ontology[i].name)
    except: 
    	return("") 



def term_enrichment(all_ids_to_annotations, group_ids, ontology, inherited=False):
    """ Obtain a dataframe with the results of a term enrichment analysis using Fisher exact test with the results sorted by p-value.
    
    Args:
        all_ids_to_annotations (dict of int:list of str): A mapping between unique integer IDs (for genes) and list of ontology term IDs annotated to them.

        group_ids (list of int): The IDs which should be a subset of the dictionary argument that refer to those belonging to the group to be tested.
        
        ontology (oats.annotation.ontology.Ontology): An ontology object that shoud match the ontology from which the annotations are drawn.
        
        inherited (bool, optional): By default this is false to indicate that the lists of ontology term IDs have not already be pre-populated to include the terms that are 
        superclasses of the terms annotated to that given ID. Set to true to indicate that these superclasses are already accounted for and the process of inheriting additional
        terms should be skipped.
    
    Returns:
        pandas.DataFrame: A dataframe sorted by p-value that contains the results of the enrichment analysis with one row per ontology term.
    """
    
    # If it has not already been performed for this data, using the ontology structure to inherit additional terms from these annotations.
    if inherited:
    	all_ids_to_inherited_annotations = all_ids_to_annotations
    else:
    	all_ids_to_inherited_annotations = {i:ontology.inherited(terms) for i,terms in all_ids_to_annotations.items()}
    

    # Find the list of all the unique ontology term IDs that appear anywhere in the annotations.
    unique_term_ids = list(set(flatten(all_ids_to_inherited_annotations.values())))
         
    # For each term, determine the total number of (gene) IDs that it is annotated to.
    num_ids_annot_with_term_t = lambda t,id_to_terms: [(t in terms) for i,terms in id_to_terms.items()].count(True) 
    term_id_to_gene_count = {t:num_ids_annot_with_term_t(t,all_ids_to_inherited_annotations) for t in unique_term_ids}
    total_num_of_genes = len(all_ids_to_inherited_annotations)
    df = pd.DataFrame(unique_term_ids, columns=["term_id"])
    df["term_label"] = df["term_id"].map(lambda x: _get_term_name(x,ontology))
    df["genes_with"] = df["term_id"].map(lambda x: term_id_to_gene_count[x])
    df["genes_without"] = total_num_of_genes-df["genes_with"] 
    
    # For each term, determine the total nubmer of (gene) IDs within the group to be tested that it is annotated to.
    num_of_genes_in_group = len(group_ids)
    ids_in_group_to_inherited_annotations = {i:terms for i,terms in all_ids_to_inherited_annotations.items() if i in group_ids}
    term_id_to_gene_in_group_count = {t:num_ids_annot_with_term_t(t,ids_in_group_to_inherited_annotations) for t in unique_term_ids}
    df["group_genes_with"] = df["term_id"].map(lambda x: term_id_to_gene_in_group_count[x])
    df["group_genes_without"] = num_of_genes_in_group-df["group_genes_with"] 
    
    # Using those values, perform the Fisher exact test to obtain a p-value for each term, sort the results, and return.
    df["p_value"] = df.apply(lambda row: fisher_exact([[row["group_genes_with"],row["genes_with"]],[row["group_genes_without"],row["genes_without"]]])[1], axis=1)
    df.sort_values(by="p_value", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return(df)










