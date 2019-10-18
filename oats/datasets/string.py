
import pandas as pd
import numpy as np
import sys




def get_stringdb_information(filename, name_to_id_dict):


	# Read in the table from the STRING database and do any necessary preprocessing.
	df = pd.read_table(filename, delim_whitespace=True)
	df["protein1"] = df["protein1"].apply(_remove_species_code)		# TODO generalize this, this only works for A. thaliana
	df["protein2"] = df["protein2"].apply(_remove_species_code)		# TODO generalize this, this only works for A. thaliana


	# Generate new columns that reflect object IDs rather than protein names from STRING.
	df["from"] = df["protein1"].map(name_to_id_dict)
	df["to"] = df["protein2"].map(name_to_id_dict)



	# The previous step introduces NaN's into the ID columns because there could be proteins mentioned
	# in the STRING database file that are not present at all in the dataset being worked with. We want
	# to remove the lines that have NaN's in them in order to return a table where the rows correspond
	# to protein to protein interactions from STRING between two proteins of genes which are present in 
	# the current dataset. However, we also want to keep track of how many proteins were in the STRING
	# data table and the current dataset but had interactions with proteins not in the current dataset.
	# These are the IDs that would be removed when removing all rows with NaN, so remember them first.
	
	ids_mentioned_in_string = pd.unique(df[["from","to"]].dropna().values.ravel('K'))
	df.dropna(axis=0, inplace=True)
	df = df[["from","to", "combined_score"]]

	# The dataframe right now is specifying directed edges rather than undirected edges.
	# We need to introduce redundancy to enforce undirected nature, so including i,j and
	# j,i edges specifically in the dataframe. This ensures that this dataframe can be 
	# merged with edgelists from elsewhere in the package without worrying about the order
	# of the genes mentioned. Drop duplicates after doing this in case both directions of
	# particular edge were already specified in the STRING dataset anyway.
	df_flipped = df[["to","from","combined_score"]]
	df_flipped.columns = ["from","to","combined_score"]
	df = pd.concat([df, df_flipped])
	df.drop_duplicates(keep="first", inplace=True)


	return(df, ids_mentioned_in_string)












# TODO generalize this method, or have lambda be passed in instead of something.
def _remove_species_code(s):
	s = s.replace("3702.", "")
	s = s.replace(".1", "")
	return(s)