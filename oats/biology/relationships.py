import pandas as pd
import numpy as np



class ProteinInteractions:

	"""
	This is a class for accessing information about relationships between genes in a dataset of interest
	by looking up these genes in the STRING protein-protein interaction database. 
	
	
	Attributes:
		name_map (TYPE): Description

		df: A dataframe containing all the STRING information but with protein names replaced with unique IDs.
	
		ids: A list of all the IDs which map to a protein which was mentioned atleast once in the STRING files passed in.

	"""


	def __init__(self, gene_to_id_dict, name_mapping_file, *string_data_files):		
		"""
		Args:
			gene_to_id_dict (dict of oats.biodata.gene.Gene:int): Mapping between gene objects and unique integer IDs from a dataset.

			name_mapping_file (str): The path to a file linking gene names with protein names used in STRING, available from STRING.

			*string_data_files (str): Any number of paths to protein-protein interaction files obtained from STRING.
		"""
		self.name_map = pd.read_table(name_mapping_file)
		self.df, self.ids = self._process_interaction_files(gene_to_id_dict, string_data_files)





	def get_df(self):
		"""Summary
		
		Returns:
			TYPE: Description
		"""
		return(self.df)



	def get_ids(self):
		"""Summary
		
		Returns:
			TYPE: Description
		"""
		return(self.ids)



	def _process_interaction_files(self, gene_to_id_dict, string_data_files):
		"""Summary
		
		Args:
			gene_to_id_dict (TYPE): Description
			string_data_files (TYPE): Description
		
		Returns:
			TYPE: Description
		"""
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














class AnyInteractions:



	"""
	This is a class for accessing information about relationships between genes in a dataset of interest
	by parsing csv files that may contain information about relationships between some or all of those genes.
	The first and second columns should contain strings that refer to gene names, and will only be used if
	those strings match strings which are given as keys in the provided dictionary. The third column should
	be a numerical value indicating a weight associated with the edge or relationsip or interaction between
	those two given genes.

	Attributes:
		df: A dataframe containing all this known information but with protein names replaced with unique IDs.
	
		ids: A list of all the IDs which map to a protein which was mentioned atleast once in the file passed in.	
	"""



	def __init__(self, name_to_id_dictionary, filename):		
		"""
		Args:
			name_to_id_dictionary (dict of str:int): Mapping between gene name strings and unique integer IDs from a dataset.
		
			filename (str): Path to a csv file containing lines that identify edges between strings mentioned in the dictionary.
		"""
		self.df,self.ids = self._get_edge_values_from_file(name_to_id_dictionary, filename)



	def get_df(self):
		"""Get a dataframe specifying relationships between IDs and their weights identified from this file.
		
		Returns:
			pandas.DataFrame: The dataframe of identified relationships.
		"""
		return(self.df)



	def get_ids(self):
		"""Get a list of all the IDs from the passed in dictionary that were successfully associated to this file. Note that there
		may be IDs in this list that are not present in the dataframe of identifed relationships, because an ID could have been 
		associated to a gene within the file, but not have relationships to any other genes that were succcessfully mapped to IDs,
		only to genes that are not mapped to an ID within the passed in dictionary.
		
		Returns:
			list: The list of IDs.
		"""
		return(self.ids)




	def _get_edge_values_from_file(self, name_to_id_dictionary, filename, ignore_case=True):
		"""
		This method is for producing a dataframe of known similarity values that can be merged
		with a dataframe representing an edgelist created from some dataset. The input is a 
		dictionary mapping gene names (strings) to integer IDs for objects (genes) in the
		dataset, and the filename that is passed in should be a file with the three columns where
		the first two are names of genes or some other object that can be mapped using the
		dictionary to particular IDs, and the third column is the similarity value that should
		be added. The file should not have a header row.
		
		Args:
			name_to_id_dictionary (dict): Mapping between gene names and IDs.
			filename (str): The file with from, to, and value columns.
		
		Returns:
			pandas.DataFrame: The dataframe of ID pairs and associated values.
		"""

		# Reading in the dataframe specified by this file.
		df = pd.read_table(filename, delim_whitespace=True, usecols=[0,1,2], header=None)


		# Account for the rows that use multiple genes names separated by a bar.
		new_row_tuples = []
		for row in df.itertuples():
			gene_1_names = row[1].split("|")
			gene_2_names = row[2].split("|")
			value = row[3]
			for g1 in gene_1_names:
				for g2 in gene_2_names:
					new_row_tuples.append((g1,g2,value))
		df = pd.DataFrame(new_row_tuples)



		# Generate new columns that reflect object IDs rather than protein names from STRING.
		if ignore_case:
			name_to_id_dictionary = {k.lower():v for k,v in name_to_id_dictionary.items()}
			df["from"] = df.iloc[:,0].str.casefold().map(name_to_id_dictionary)
			df["to"] = df.iloc[:,1].str.casefold().map(name_to_id_dictionary)
			df["value"] = df.iloc[:,2]
		else:
			df["from"] = df.iloc[:,0].map(name_to_id_dictionary)
			df["to"] = df.iloc[:,1].map(name_to_id_dictionary)
			df["value"] = df.iloc[:,2]



		# The previous step introduces NaN's into the ID columns because there could be proteins mentioned
		# in the STRING database file that are not present at all in the dataset being worked with. We want
		# to remove the lines that have NaN's in them in order to return a table where the rows correspond
		# to protein to protein interactions from STRING between two proteins of genes which are present in 
		# the current dataset. However, we also want to keep track of how many proteins were in the STRING
		# data table and the current dataset but had interactions with proteins not in the current dataset.
		# These are the IDs that would be removed when removing all rows with NaN, so remember them first.
		l1 = list(pd.unique(df[["from"]].dropna().values.ravel('K')))
		l2 = list(pd.unique(df[["to"]].dropna().values.ravel('K')))
		ids_mentioned_in_this_file = list(set(l1+l2))


		df.dropna(axis=0, inplace=True)
		df = df[["from","to", "value"]]

		# The dataframe right now is specifying directed edges rather than undirected edges.
		# We need to introduce redundancy to enforce undirected nature, so including i,j and
		# j,i edges specifically in the dataframe. This ensures that this dataframe can be 
		# merged with edgelists from elsewhere in the package without worrying about the order
		# of the genes mentioned. Drop duplicates after doing this in case both directions of
		# particular edge were already specified in the STRING dataset anyway.
		df_flipped = df[["to","from","value"]]
		df_flipped.columns = ["from","to","value"]
		df = pd.concat([df, df_flipped])
		df.drop_duplicates(keep="first", inplace=True)
		return(df, ids_mentioned_in_this_file)







