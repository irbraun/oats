from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from functools import reduce
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier








def combine_dfs_with_name_dict(dfs_dict):
	"""Produce a dataframe in a shape that can be used to train models.
	Args:
	    dfs_dict (dict): Mapping from name strings to dataframe objects.
	Returns:
	    pandas.DataFrame: A single dataframe where each similarity column
	    in the original dataframes has become a new column in the combined
	    dataframe where the header for that column is the name that was in
	    the dictionary. This way each individual passed in dictionary can
	    be used to collect named features for a new dataframe that can be 
	    used to train models for predicting the best combination of features.
	"""
	for name,df in dfs_dict.items():
		df.rename(columns={"similarity":name}, inplace=True)
	_verify_dfs_are_consistent(*dfs_dict.values())
	merged_df = reduce(lambda left,right: pd.merge(left,right,on=["from","to"], how="outer"), dfs_dict.values())
	merged_df.fillna(0.000, inplace=True)
	return(merged_df)







def subset_df_with_ids(df, ids):
	df = df[df["from"].isin(ids) & df["to"].isin(ids)]
	return(df)













def _verify_dfs_are_consistent(*similarity_dfs):
	"""Check that each dataframe specifies the same set of edges.
	Args:
	    *similarity_dfs: Any number of dataframe arguments.
	Raises:
	    Error: The dataframes were found to not all be describing the same graph.
	"""
	id_sets = [set() for i in range(0,len(similarity_dfs))]
	for i in range(0,len(similarity_dfs)):
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["from"].values)))
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["to"].values)))
	for (s1, s2) in list(itertools.combinations_with_replacement(id_sets, 2)):	
		if not len(s1.difference(s2)) == 0:
			raise ValueError("Dataframes specifying networks are not consisent.")




