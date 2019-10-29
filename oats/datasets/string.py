
import pandas as pd
import numpy as np
import sys





class String:

	"""
	Attributes
	df: A dataframe containing all the STRING information but with protein names replaced with unique IDs.
	ids: A list of all the IDs which map to a protein which was mentioned atleast once in the STRING files passed in.
	"""


	def __init__(self, gene_to_id_dict, name_mapping_file, *string_data_files):		
		self.name_map = pd.read_table(name_mapping_file)
		self.df, self.ids = self.process_interaction_files(gene_to_id_dict, string_data_files)



	def process_interaction_files(self, gene_to_id_dict, string_data_files):

		# Need to produce a mapping between STRING names and our IDs. Having this will allow for replacing
		# the protein names that are in the STRING database files with IDs that are relevant to this data-
		# set. Do this by iterating through the IDs used and checking for appropriate match between the 
		# names associated with those genes for that species and something from the file of all names 
		# associated with any protein in the STRING database.
		species_name_to_ncbi_taxid_mapping = {
			"ath":3702, 
			"zma":4577,
			"sly":4081,
			"mtr":3880,
			"osa":4530,
			"gmx":3847} # Update this line when adding new species.
		self.name_map = self.name_map.loc[list(species_name_to_ncbi_taxid_mapping.values())]
		string_name_to_id_dict = {}
		for identifier,gene in gene_to_id_dict.items():
			string_names = []
			species = species_name_to_ncbi_taxid_mapping[gene.species]
			for name in gene.names:
				try:
					string_names.extend(self.name_map.loc[(species, name)].values)
				except KeyError:
					pass
			for string_name in string_names:
				string_name_to_id_dict[string_name] = identifier


		# Read in the tables from the STRING database and do any necessary preprocessing.
		dfs = [pd.read_table(filename, delim_whitespace=True) for filename in string_data_files]
		df = pd.concat(dfs)


		# Generate new columns that reflect object IDs rather than protein names from STRING.
		df["from"] = df["protein1"].map(string_name_to_id_dict)
		df["to"] = df["protein2"].map(string_name_to_id_dict)

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







