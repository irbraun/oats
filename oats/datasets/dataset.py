from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from pandas import DataFrame
import os
import sys
import glob


from oats.datasets.gene import Gene
from oats.nlp.preprocess import concatenate_descriptions
from oats.nlp.preprocess import concatenate_with_bar_delim





class Dataset:


	def __init__(self):
		self.col_names = ["id", "species", "gene_names", "description", "term_ids", "pmid"]
		self.col_names_without_id = self.col_names
		self.col_names_without_id.remove("id")
		self.df = pd.DataFrame(columns=self.col_names)



	def _reset_ids(self):
		"""
		This method is called after every preprocessing, subsampling, sorting, or merging step
		done during the preprocessing and creation of a given dataset. It resets the actual row
		indices of the internal dataframe object, forgets the old ones, and uses the new ones 
		to populate a column to use as IDs for entries in the dataset. This ensures that there
		will always be a ID value to access where the IDs are always unique integers.
		"""
		self.df.reset_index(drop=True,inplace=True)
		self.df.id = [int(i) for i in self.df.index.values]
		self.df = self.df[["id", "species", "gene_names", "description", "term_ids", "pmid"]]






	def add_data(self, df_newlines):
		"""
		Add a dataframe containing information to this dataset. The dataframe provided has to 
		contain all the columns mentioned in the constructur for this class except for the ID
		column which is generated once all the data has been added so that the values are always
		still unique and useful as matrix indices (integers from 0 to n) after any data is added
		to the dataset.
		Args:
		    df_newlines (pandas.DataFrame): The dataframe containing rows to be added.
		"""
		df_newlines = df_newlines[self.col_names_without_id]
		df_newlines.loc[:,"id"] = None
		df_newlines.fillna("", inplace=True)
		df_newlines.loc[:,"pmid"] = str(df_newlines["pmid"])
		self.df = self.df.append(df_newlines, ignore_index=True, sort=False)
		self.df = self.df.drop_duplicates(keep="first", inplace=False)
		self._reset_ids()



	def filter_random_k(self, k, seed):
		"""Remove all but k randomly sampled points from the dataset.
		Args:
		    k (int): The number of datapoints or rows to retain.
		    seed (int): Seed value for reproducibility of random process.
		"""
		self.df = self.df.sample(n=k, random_state=seed)
		self._reset_ids()



	def filter_has_description(self):
		"""Remove points that don't have a text description.
		"""
		self.df = self.df[self.df["description"] != ""] 
		self._reset_ids()


	def filter_has_annotation(self):
		"""Remove points that don't have atleast one ontology term annotation.
		"""
		self.df = self.df[self.df["term_ids"] != ""]
		self._reset_ids()



	def filter_with_ids(self, ids):
		"""Retain only points that have IDs in the list. 
		"""
		self.df = self.df[self.df["id"].isin(ids)]
		self._reset_ids()





	





	def get_gene_dictionary(self):
		"""Get a mapping from IDs to gene objects.
		Returns:
		    TYPE: Description
		"""
		gene_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			gene_names = row.gene_names.split(delim)
			gene_obj = Gene(names=gene_names, species=row.species)
			gene_dict[row.id] = gene_obj
		return(gene_dict)



	def get_annotations_dictionary(self):
		"""Get a mapping from IDs to lists of ontology term IDs.
		Returns:
		    TYPE: Description
		"""
		annotations_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			term_ids = row.term_ids.split(delim)
			annotations_dict[row.id] = term_ids
		return(annotations_dict)



	def get_description_dictionary(self):
		"""Get a mapping from IDs to text descriptions.
		Returns:
		    TYPE: Description
		"""
		description_dict = {identifier:description for (identifier,description) in zip(self.df.id, self.df.description)}
		return(description_dict)








	def collapse_by_first_gene_name(self):
		"""
		This method reorganizes the internal dataframe object so that any lines that were referring
		to the same species and where the first value in the list of gene names is identical for 
		the entry are merged. Text descriptions are concatenated and a union of the gene names, term
		IDs and references are retained. This is much faster than collapsing based on all the gene
		names because it can use the groupby function because it's looking at a single value rather
		than an overlap in a list of values. 
		"""

		# Create the column that can be used to group by.
		self.df["first_gene_name"] = self.df["gene_names"].apply(lambda x: x.split("|")[0])
		self.df["first_gene_name"] = self.df["first_gene_name"]+":"+self.df["species"]

		# Groupy by that column and merge the other fields appropriately.
		collapsed_df = self.df.groupby("first_gene_name").agg({"species": lambda x: x.values[0],
																"gene_names": lambda x: concatenate_with_bar_delim(*x),
																"description":lambda x: concatenate_descriptions(*x),
																"term_ids":lambda x: concatenate_with_bar_delim(*x),
																"pmid":lambda x: concatenate_with_bar_delim(*x)})
		collapsed_df["id"] = None
		self.df = collapsed_df
		self._reset_ids()











	def collapse_by_all_gene_names(self):
		"""
		This method reorganizes the internal dataframe object so that any lines that were referring
		to the same species and that have an overlap of atleast one in the gene names that were included
		for those lines are combined. This is slow. All data is retained, so that a union of the gene 
		names and union of ontology term annotations are found for any lines combined. Text descriptions
		are combined as well by concatenating using the supporting methods.
		"""

		# Only perform this operation on slices of the data for one species at a time.
		# This Enforces that genes with the same name across two species have to be two seperate nodes.
		# Nodes that correspond to different species can never be merged.
		num_new_rows = 0
		for species in pd.unique(self.df.species):
			print(species)


			# (1) Create a mapping from gene name strings to row indices where that string is mentioned.
			gene_mention_map = defaultdict(list)
			for row in self.df.itertuples():
				if row.species == species:
					delim = "|"
					gene_names = row.gene_names.split(delim)
					for gene_name in gene_names:
						gene_mention_map[gene_name].append(row.Index)


			# (2) Create a list of sets where all indices in a given set contain synonymous genes (overlap in >=1 name used).
			list_of_sets_of_row_indices = []
			for gene_name in gene_mention_map.keys():

				# Determine which existing synonymous set this gene belongs to, if any.
				set_index_where_this_gene_belongs = -1
				i = 0
				for set_of_row_indices in list_of_sets_of_row_indices:
					if len(set(gene_mention_map[gene_name]).intersection(set_of_row_indices)) > 0:
						set_index_where_this_gene_belongs = i
						list_of_sets_of_row_indices[i].update(gene_mention_map[gene_name])
						break
					i = i+1

				# If this gene doesn't belong in any of those sets, start a new set with all it's corresponding row indices.
				if set_index_where_this_gene_belongs == -1:
					list_of_sets_of_row_indices.append(set(gene_mention_map[gene_name]))


			# (3) Add rows which contain merged information from multiple rows where the same gene was mentioned.
			num_new_rows = num_new_rows + len(list_of_sets_of_row_indices)
			for set_of_row_indices in list_of_sets_of_row_indices:
				relevant_rows = self.df.iloc[list(set_of_row_indices)]
				description = concatenate_descriptions(*relevant_rows.description.tolist())
				gene_names = concatenate_with_bar_delim(*relevant_rows.gene_names.tolist())
				term_ids = concatenate_with_bar_delim(*relevant_rows.term_ids.tolist())
				pmids = concatenate_with_bar_delim(*relevant_rows.pmid.tolist())
				new_row = {
					"id":None,
					"species":species,
					"gene_names":gene_names,
					"description":description,
					"term_ids":term_ids,
					"pmid":pmids,
				}
				self.df.append(new_row, ignore_index=True, sort=False)


		# Retain only the newly added rows, reset the ID values for each row that correspond to one node.
		self.df = self.df.iloc[-num_new_rows:]
		self._reset_ids()





	def sort_by_species(self):
		self.df.sort_values(by=["species"], inplace=True)
		self._reset_ids()



	def write_to_csv(self, path):
		self.sort_by_species()
		self.df.to_csv(path, index=False)




	def describe(self):
		print("Number of rows in the dataframe: {}".format(len(self.df)))
		print("Number of unique IDs:            {}".format(len(pd.unique(self.df.id))))
		print("Number of unique descriptions:   {}".format(len(pd.unique(self.df.description))))
		print("Number of unique gene name sets: {}".format(len(pd.unique(self.df.gene_names))))
		print("Number of species represented:   {}".format(len(pd.unique(self.df.species))))




