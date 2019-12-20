import pandas as pd
import numpy as np
import sys



class Edges:

	"""
	Attributes
	df: A dataframe containing all this known information but with protein names replaced with unique IDs.
	ids: A list of all the IDs which map to a protein which was mentioned atleast once in the file passed in.
	"""




	def __init__(self, name_to_id_dictionary, filename):		

		self.df,self.ids = self._get_edge_values_from_file(name_to_id_dictionary, filename)



	def _get_edge_values_from_file(self, name_to_id_dictionary, filename):
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
		df = pd.read_table(filename, delim_whitespace=True, usecols=[0,1,2])

		# Generate new columns that reflect object IDs rather than protein names from STRING.
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
		ids_mentioned_in_this_file = pd.unique(df[["from","to"]].dropna().values.ravel('K'))
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


