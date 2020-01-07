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
		self.col_names = ["id", "species", "gene_names", "description", "term_ids"]
		self.col_names_without_id = self.col_names
		self.col_names_without_id.remove("id")
		self.df = pd.DataFrame(columns=self.col_names)







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
		self.df = self.df.append(df_newlines, ignore_index=True, sort=False)
		self.df = self.df.drop_duplicates(keep="first", inplace=False)
		self._reset_ids()






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
		self.df = self.df[["id", "species", "gene_names", "description", "term_ids"]]











	####### Filtering the dataset destsructively ##########



	def filter_random_k(self, k, seed=7919):
		"""Remove all but k randomly sampled points from the dataset.
		"""
		self.df = self.df.sample(n=k, random_state=seed)


	def filter_by_species(self, *species):
		"""Retain only points that are for species in this list.
		"""
		self.df = self.df[self.df["species"].isin(species)]


	def filter_has_description(self):
		"""Remove points that don't have a text description.
		"""
		self.df = self.df[self.df["description"] != ""] 


	def filter_has_annotation(self, ontology_name=None):
		"""
		Remove points that don't have atleast one ontology term annotation,
		or atleast one term annotation from the provided ontology if that 
		argument is used.
		"""
		if ontology_name is not None:
			self.df = self.df[self.df["term_ids"].str.contains(ontology_name)]
		else:
			self.df = self.df[self.df["term_ids"] != ""]

	def filter_with_ids(self, ids):

		"""Retain only points that have IDs in the list. 
		"""
		self.df = self.df[self.df["id"].isin(ids)]













	####### Accessing information from the dataset ##########


	def get_gene_dictionary(self):
		"""Get a mapping from IDs to gene objects.
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
		"""
		annotations_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			term_ids = row.term_ids.split(delim)
			annotations_dict[row.id] = term_ids
		return(annotations_dict)


	def get_description_dictionary(self):
		"""Get a mapping from IDs to text descriptions.
		"""
		description_dict = {identifier:description for (identifier,description) in zip(self.df.id, self.df.description)}
		return(description_dict)


	def get_species_dictionary(self):
		"""Get a mapping from IDs to species strings.
		"""
		species_dict = {identifier:species for (identifier,species) in zip(self.df.id, self.df.species)}
		return(species_dict)



	def get_ids(self):
		"""Get a list of the IDs for all entries in this dataset.
		"""
		return(list(self.df.id.values))



	def get_species(self):
		"""Get a list of all the species that are represented in this dataset.
		"""
		return(list(pd.unique(self.df.species.values)))



	def get_name_to_id_dictionary(self, unambiguous=True):
		"""
		Get a mapping between gene names and the object ID they refer to. When the unambiguous
		parameter is True then the only keys that are present in the dictionary are names that
		map to a single gene from the dataset, so this excludes names that map to genes from
		multiple species. This is because this method is useful for mapping gene names (with
		no species information) in other files to IDs from this dataset. So those files should
		only be used when they contain unambiguous names or accessions that encode the species
		information. 

		TODO create version of this that uses both the species string or number and the gene
		names as the key to the generated dictionary, so that unambiguous parameter can be 
		False and genes with the same name from two species can be mapped to the correct IDs.

		"""
		name_to_id_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			names = row.gene_names.split(delim)
			for name in names:
				if unambiguous and name not in name_to_id_dict:
					name_to_id_dict[name] = row.id
				else:
					name_to_id_dict[name] = row.id
		return(name_to_id_dict) 












	######## Merging rows in the dataset based on overlaps in gene names #########


	def collapse_by_first_gene_name(self):
		"""
		This method reorganizes the internal dataframe object so that any lines that were referring
		to the same species and where the first value in the list of gene names is identical for 
		the entry are merged. Text descriptions are concatenated and a union of the gene names, term
		IDs and references are retained. This is not appreciably faster than collapsing by all the 
		gene names instead of just the first one, because that problem can be formulated as solving
		the connected components problem before doing the group by step. So this method is included
		not to save time but for cases where only the first gene name is unique gene identifer and 
		the other are potentially not.
		"""

		# Create the column that can be used to group by.
		self.df["first_gene_name"] = self.df["gene_names"].apply(lambda x: x.split("|")[0])
		self.df["first_gene_name"] = self.df["first_gene_name"]+":"+self.df["species"]

		# Groupy by that column and merge the other fields appropriately.
		collapsed_df = self.df.groupby("first_gene_name").agg({"species": lambda x: x.values[0],
																"gene_names": lambda x: concatenate_with_bar_delim(*x),
																"description":lambda x: concatenate_descriptions(*x),
																"term_ids":lambda x: concatenate_with_bar_delim(*x)})
		collapsed_df["id"] = None
		self.df = collapsed_df
		self._reset_ids()


	@staticmethod
	def generate_edges(row):
		"""
		Given a row in a dataframe that matches the column signature of the one used here, generate
		a list of edges between the entry's ID and each of the gene names mentioned for that entry.
		This uses a representation of the gene names that includes the species as a prefix so that
		the same gene name used for two different species are not considered the same gene. This is 
		treating the gene names and ID's as nodes in a graph for the purpose of collapsing the dataset
		as a connected components problem, see the description in the function to collapse by all 
		gene names below.
		"""
		names = row["gene_names"].split("|")
		edges = [(row["id"],"{}.{}".format(row["species"],name)) for name in names]
		return(edges)



	def collapse_by_all_gene_names(self):
		"""
		The input is the dataframe that has the form:

		ID  Names
		1	A
		2	A,B
		3	B,C,D
		4   E,F

		The behavior of this function should be to add an additional column that has one unique value
		for all the rows that, based on the assumption that all values in 'Names' are unique identifers,
		those rows all represent the same actual gene. Note that this might not be a good assumption for 
		some of the sources of gene synonyms in some of the datasets used, so only values that are likely
		unique such as true accessions of gene model names should be included in the column used as the 
		gene names column here. The result of adding a new component column would be as follows.

		ID  Names    Component
		1	A        1
		2	A,B      1
		3	B,C,D    1
		4   E,F      2

		Then the dataframe can be aggregated based on the values in the component column to yield the 
		following smaller version of the dataframe, with reset ID values to reflect the componets.

		ID  Names    Component
		1	A,B,C,D  1
		2   E,F      2

		This is solved by treating the ID values and the name values as unique node names in a graph, 
		and then given the set of edges that are specifed in that dataframe (1-A, 1-B, ..., 4-F), the
		problem can be solved as a connected components problem.
		"""

		import networkx as nx
		from itertools import chain

		# Build the graph model of this data where nodes are gene names or IDs.
		g = nx.Graph()
		edges = self.df.apply(Dataset.generate_edges, axis=1)
		edges = list(chain.from_iterable(edges.values))
		g.add_edges_from(edges)

		# Get the connected components of the graph, a list of lists of nodes. Each component will
		# always have one or more ID's in it as (a) node(s), because every gene name has to be
		# associated with atleast one ID, corresponding to the row that it was in in the input data.
		components = nx.connected_components(g)

		# Build a dictionary mapping node names (IDs and gene names) to component values.
		# Is there a more efficient way to get a node to component mapping? There probably
		# should be we don't need everything to be in the mapping, just one thing from each 
		# entry, which could be just the ID values, don't need any of the gene names. This is
		# because all node mentioned in one entry will always be in the same component. One 
		# solution would be to sort all the lists of nodes then just take the first, if we
		# could make sure the ID value would always be first? That might be even slower, 
		# depends on ratio of number of entries to number of names?
		node_to_component = {}
		component_index = 0
		for node_set in components:
			for node in node_set:
				node_to_component[node] = component_index
			component_index = component_index+1

		# Create a new column that indicates which connected component that entry maps to.
		self.df["component"] = self.df["id"].map(node_to_component)

		# Groupy by the connected component column and merge the other fields appropriately.
		self.df = self.df.groupby("component").agg({"species": lambda x: x.values[0],
															"gene_names": lambda x: concatenate_with_bar_delim(*x),
															"description":lambda x: concatenate_descriptions(*x),
															"term_ids":lambda x: concatenate_with_bar_delim(*x)})
		
		# Reset the ID values in the dataset to reflect this change.
		self.df["id"] = None
		self._reset_ids()





		







	######## Summarizing, saving, or converting the whole dataset #########

	def _sort_by_species(self):
		self.df.sort_values(by=["species"], inplace=True)
		self._reset_ids()

	def to_csv(self, path):
		self._sort_by_species()
		self.df.to_csv(path, index=False)

	def to_pandas(self):
		self._sort_by_species()
		return(self.df)

	def describe(self):
		print("Number of rows in the dataframe: {}".format(len(self.df)))
		print("Number of unique IDs:            {}".format(len(pd.unique(self.df.id))))
		print("Number of unique descriptions:   {}".format(len(pd.unique(self.df.description))))
		print("Number of unique gene name sets: {}".format(len(pd.unique(self.df.gene_names))))
		print("Number of species represented:   {}".format(len(pd.unique(self.df.species))))







