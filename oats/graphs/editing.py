import numpy as np
import pandas as pd
import functools

from oats.utils.utils import flatten






def merge_edgelists(dfs_dict, default_value=None):	
	""" 
	Takes a dictionary mapping between names and {from,to,value} formatted dataframes and
	returns a single dataframe with the same nodes listed but where there is now one value
	column for each dataframe provided, with the name of the column being the corresponding
	name.

	Args:
		dfs_dict (dict): Mapping between strings (names) and pandas.DataFrame objects.
		default_value (None, optional): A value to be inserted where none is present. 
	
	Returns:
		TYPE: Description
	"""
	for name,df in dfs_dict.items():
		df.rename(columns={"value":name}, inplace=True)
	merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on=["from","to"], how="outer"), dfs_dict.values())
	if not default_value == None:
		merged_df.fillna(default_value, inplace=True)
	return(merged_df)








def make_undirected(df):
	"""
	The dataframe passed in must be in the form {from, to, [other...]}.
	Convert the undirected edgelist where an edge (j,i) is always implied by an edge (i,j) to a directed edgelist where
	both the (i,j) and (j,i) edges are explicity present in the dataframe. This is done so that we can make us of the
	groupby function to obtain all groups that contain all edges between some given node and everything its mapped to 
	by just grouping base on one of the columns specifying a node. This is easier than using a multi-indexed dataframe.
	
	Args:
	    df (pandas.DataFrame): Any dataframe with in the form {from, to, [other...]}.
	
	Returns:
	    pandas.DataFrame: The updated dataframe that includes edges in both directions.
	"""
	other_columns = df.columns[2:]
	flipped_edges = df[flatten(["to","from",other_columns])]      # Create the flipped duplicate dataframe.
	flipped_edges.columns = flatten(["from","to",other_columns])  # Rename the columns so it will stack correctly
	df = pd.concat([df, flipped_edges])
	df.drop_duplicates(keep="first", inplace=True)
	return(df)
	








def remove_self_loops(df):
	""" 
	Removes all edges connecting the same ID.
	"""
	return(df[df["from"] != df["to"]])








def subset_edgelist_with_ids(df, ids):
	""" 
	Removes all edges from an edgelist that connects two nodes where one or two of the
	nodes are not present in the passed in list of nodes to retain. This results in an
	edge list that specifies a subgraph of the one specified by the original edgelist.

	Args:
		df (pandas.DataFrame): The edge list before subsetting.
		ids (list): A list of the node IDs that are the only ones left after subsetting.

	Returns:
		pandas.DataFrame: The edge list after subsetting.
	"""
	df = df[df["from"].isin(ids) & df["to"].isin(ids)]
	return(df)




