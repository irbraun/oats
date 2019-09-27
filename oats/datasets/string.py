
import pandas as pd
import numpy as np











def get_string_protein_links_df(filename, genes):
	""" 
	Generates a dataframe specifying protein to protein interactions from STRING that
	includes columns that have the IDs from the objects in the passed in genes dictionary
	that map to protein names from the STRING dataset. This way the returned dataframe is
	ready to be merged with  
	
	Args:
	    filename (str): Path to the database protein interaction file.
	    genes (dict): Mapping between object IDs and gene objects.
	
	Returns:
	    pandas.DataFrame: All the protein interactions relevant to the passed in dataset.
	"""



	# Build the dataframe that is just reading in directly the STRING database file.
	df = pd.read_table(filename, delim_whitespace=True)

	# TODO find a better way to do this that would be applicable to all the species?
	# Clean the protein names so that they will be compatible with how those genes are defined in the dataset.
	df["protein1"] = df["protein1"].apply(_remove_species_code)
	df["protein2"] = df["protein2"].apply(_remove_species_code)
	
	# Add columns that refer to internal gene IDs that are specific to the genes dictionary.
	protein_name_to_id_dict = _get_protein_to_id_mapping(df, genes)
	df["from"] = df["protein1"].apply(lambda x: protein_name_to_id_dict[x])
	df["to"] = df["protein2"].apply(lambda x: protein_name_to_id_dict[x])

	# The previous step introduces NAs in the ID columns where no mapping was found from the protein name
	# to something in the dictionary of gene objects. Since it's assumed that the only information we're 
	# working with is the genes that are passed in, drop all those rows because they don't contain relevant
	# information as those proteins aren't mapped to anything else in the dataset.
	df.dropna(axis=0, inplace=True)

	# Only interested in keeping a subset of these columsn, and put them in a more logical order.
	columns = ["from","to", "combined_score"]
	df = df[columns]
	return(df)








# TODO generalize this method, or have lambda be passed in instead of something.
def _remove_species_code(s):
    s = s.replace("3702.", "")
    s = s.replace(".1", "")
    return(s)






def _get_protein_to_id_mapping(df, genes):
    string_protein_name_to_id = {}
    string_protein_names = set()
    string_protein_names.update(pd.unique(df.protein1))
    string_protein_names.update(pd.unique(df.protein2))
    string_protein_names = list(string_protein_names)

    for protein_name in string_protein_names:
    	matches = set()
    	for gene_id,gene_obj in genes.items():
    		if protein_name in gene_obj.names:
    			matches.add(gene_id)
    	matches = list(matches)

    	if len(matches) == 0:
    		string_protein_name_to_id[protein_name] = None
    	elif len(matches) == 1:
    		string_protein_name_to_id[protein_name] = matches[0]
    	else:
    		raise KeyError("The STRING protein with name {} matched multiple genes: {}".format(protein_name, " ".join(matches)))

    return(string_protein_name_to_id)








