from itertools import chain
from collections import defaultdict
import pandas as pd
import itertools
import networkx as nx

from oats.biology.gene import Gene
from oats.nlp.preprocess import concatenate_texts, concatenate_with_delim





class Dataset:

	"""A class that wraps a dataframe containing gene names, text, and ontology annotations.
	
	Attributes:
		df (pandas.DataFrame): The dataframe containing the dataset accessed through this class or the path
		to a csv file that can be loaded as a comparable dataframe. 

	"""
	





	def __init__(self, data=None, keep_ids=False, case_sensitive=False, source=None):
		"""
		Args:
		    data (pandas.DataFrame or str, optional): A dataframe containing the data to be added to this
		    dataset, or the path to a csv file that can be loaded as a comparable dataframe. The columns
		    of this dataframe must contain "species", "gene_names", "gene_synonyms", "description", "term_ids",
		    and "sources". Any of those columns can contain any number of missing values but the columns must 
		    exist. Any columns that are outside of that list of column names are ignored. Gene names, symbols,
		    and ontology term IDS in the "gene_names", "gene_synonyms", and "term_ids" columns should be pipe 
			delimited.

		"""

		all_column_names = ["id", "species", "unique_gene_identifiers", "other_gene_identifiers", "gene_models", "descriptions", "annotations", "sources"]
		column_names_without_id = all_column_names[1:]
		self._col_names = all_column_names
		self._col_names_without_id = column_names_without_id
		self.df = pd.DataFrame(columns=self._col_names)
		if data is not None:
			if keep_ids:
				self._add_data_with_ids(new_data=data)
			else:
				self.add_data(new_data=data, case_sensitive=case_sensitive, source=source)
		self._update_dictionaries()







	def _add_data_with_ids(self, new_data, source=None):
		"""
		Only called by the constructor. Allows for saving a dataset to a CSV then regenerating it with the same IDs.
		Retaining IDs is not supported when adding any new data, they always get merged and reset. This only works
		for creating some dataset, saving it, then reading in that exact same dataset somewhere else. This is important
		for sharing a given dataset with the streamlit application for intance. This does not call the method for 
		resetting the ID column.

		"""

		if isinstance(new_data, pd.DataFrame):
			new_data = new_data[self._col_names]
			new_data.fillna("", inplace=True)
			#new_data["descriptions"] = new_data["descriptions"].map(lambda x: x.replace(";","."))
			self.df = self.df.append(new_data, ignore_index=True, sort=False)
			self._update_dictionaries()

		elif isinstance(new_data, str):
			new_data = pd.read_csv(new_data)
			new_data = new_data[self._col_names]
			new_data.fillna("", inplace=True)
			#new_data["descriptions"] = new_data["descriptions"].map(lambda x: x.replace(";","."))
			self.df = self.df.append(new_data, ignore_index=True, sort=False)
			self._update_dictionaries()

		else:
			raise ValueError("the data argument should be filename string or a pandas dataframe object")









	def add_data(self, new_data, case_sensitive=False, source=None):
		"""Add additional data to this dataset.
		
		Args:
			new_data (pandas.DataFrame or str): A dataframe containing the data to be added to this 
			dataset, or the path to a csv file that can be loaded as a comparable dataframe. The columns 
			of this dataframe must contain "species", "gene_names", "gene_synonyms", "description", "term_ids", 
			and "sources". Any of those columns can contain any number of missing values but the columns must 
			exist. Any columns that are outside of that list of column names are ignored. Gene names, symbols,
			and ontology term IDS in the "gene_names", "gene_synonyms", and "term_ids" columns should be pipe 
			delimited.
		
		"""

		if isinstance(new_data, pd.DataFrame):
			new_data = new_data[self._col_names_without_id]
			new_data.loc[:,"id"] = None
			new_data.fillna("", inplace=True)
			#new_data["descriptions"] = new_data["descriptions"].map(lambda x: x.replace(";","."))
			self.df = self.df.append(new_data, ignore_index=True, sort=False)
			self.df = self.df.drop_duplicates(keep="first", inplace=False)
			# The IDs need to be reset before collapsing by gene names, because it makes use of the that column.
			self._reset_ids()
			self._collapse_by_all_gene_names(case_sensitive=case_sensitive)
			# These two methods are already called after collapsing by all the gene names.
			# Calling them again shouldn't break anything, it's just waste of processing.
			#self._reset_ids()
			#self._update_dictionaries()


		elif isinstance(new_data, str):
			new_data = pd.read_csv(new_data)
			new_data = new_data[self._col_names_without_id]
			new_data.loc[:,"id"] = None
			new_data.fillna("", inplace=True)
			#new_data["descriptions"] = new_data["descriptions"].map(lambda x: x.replace(";","."))
			self.df = self.df.append(new_data, ignore_index=True, sort=False)
			self.df = self.df.drop_duplicates(keep="first", inplace=False)
			# The IDs need to be reset before collapsing by gene names, because it makes use of the that column.
			self._reset_ids()
			self._collapse_by_all_gene_names(case_sensitive=case_sensitive)
			# These two methods are already called after collapsing by all the gene names.
			# Calling them again shouldn't break anything, it's just waste of processing.
			#self._reset_ids()
			#self._update_dictionaries()

		else:
			raise ValueError("the data argument should be filename string or a pandas dataframe object")






	def _reset_ids(self):
		"""
		This method is only called after initially reading in a datafile that does not already have IDs 
		speciied, or after adding additional data, or after merging based on gene identifers. It does 
		not run after filtering, so filtering can be done without destroying the mapping between existing
		IDs and the information each represents.
		"""
		self.df.reset_index(drop=True,inplace=True)
		self.df.id = [int(i) for i in self.df.index.values]
		self.df = self.df[self._col_names]










	####### Filtering the dataset destsructively ##########



	def filter_random_k(self, k, seed=1483):
		"""Remove all but k randomly sampled records from the dataset.
		
		Args:
			k (int): The number of records (IDs) to retain.

			seed (int, optional): A seed value for the random subsampling.
		"""
		self.df = self.df.sample(n=k, random_state=seed)
		self._update_dictionaries()



	def filter_by_species(self, species):
		"""Remove all records not related to these species.
		
		Args:
		    species (list of str): A list of strings referring the species names.
		"""
		self.df = self.df[self.df["species"].isin(species)]
		self._update_dictionaries()





	def filter_has_description(self):
		"""Remove all records that don't have a related text description.
		"""
		self.df = self.df[self.df["descriptions"] != ""] 
		self._update_dictionaries()





	def filter_has_annotation(self, ontology_name=None):
		"""Remove all records that don't have atleast one related ontology term annotation.
		
		Args:
			ontology_name (str, optional): A string which is the name of an ontology (e.g, "PATO", "PO").
			If this ontology name is provided then only annotations from that ontology are considered when
			filtering the dataset.
		"""
		if ontology_name is not None:
			self.df = self.df[self.df["annotations"].str.contains(ontology_name)]
		else:
			self.df = self.df[self.df["annotations"] != ""]
		self._update_dictionaries()




	def filter_with_ids(self, ids):
		"""Remove all records with IDs not in the provided list.
		
		Args:
			ids (list of int): A list of the unique integer IDs for the records to retain.
		"""
		self.df = self.df[self.df["id"].isin(ids)]
		self._update_dictionaries()














	####### Generating the dictionaries that are useful for accessing this data ##########

	def _update_dictionaries(self):
		self._species_to_name_to_ids_dictionary_with_synonyms = self._make_species_to_name_to_ids_dictionary(nonuniques=True, lowercase=False)
		self._species_to_name_to_ids_dictionary_without_synonyms = self._make_species_to_name_to_ids_dictionary(nonuniques=False, lowercase=False)
		self._species_to_name_to_ids_dictionary_with_synonyms_lowercased = self._make_species_to_name_to_ids_dictionary(nonuniques=True, lowercase=True)
		self._species_to_name_to_ids_dictionary_without_synonyms_lowercased = self._make_species_to_name_to_ids_dictionary(nonuniques=False, lowercase=True)
		self._gene_object_dictionary = self._make_gene_dictionary()






	def _make_species_to_name_to_ids_dictionary(self, nonuniques=False, lowercase=False):
		"""Summary
		
		Args:
		    nonuniques (bool, optional): Description
		    
		    lowercase (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		species_to_name_to_ids_dictionary = defaultdict(lambda: defaultdict(list))
		for row in self.df.itertuples():
			delim = "|"
			names = row.unique_gene_identifiers.split(delim)
			if nonuniques:
				names.extend(row.other_gene_identifiers.split(delim))
			if lowercase:
				names = [name.lower() for name in names]
			for name in names:
				species_to_name_to_ids_dictionary[row.species][name].append(row.id)
		return(species_to_name_to_ids_dictionary)






	def _make_gene_dictionary(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		gene_dict = {}
		for row in self.df.itertuples():
			delim = "|"

			# Parse the different types of identifiers into lists, making sure to create empty lists if there's nothing there, NOT lists with empty strings.
			species = row.species
			unique_identifiers = row.unique_gene_identifiers.split(delim)
			other_identifiers = [x for x in row.other_gene_identifiers.split(delim) if x != ""]
			gene_models = [x for x in row.gene_models.split(delim) if x != ""]
			gene_obj = Gene(species, unique_identifiers, other_identifiers, gene_models)
			gene_dict[row.id] = gene_obj
		return(gene_dict)




















	####### Accessing information from the dataset ##########


	def get_gene_dictionary(self):
		"""Get a mapping from record IDs to their corresponding gene objects.
		
		Returns:
			dict of int:oats.datasets.gene.Gene: Mapping from record IDs to gene objects.
		"""
		return(self._gene_object_dictionary)




	def get_annotations_dictionary(self, ontology_name=None):
		"""Get a mapping from IDs to lists of ontology term IDs.
		
		Returns:
		    dict of int:list of str: Mapping between record IDs and lists of ontology term IDs.
		
		Args:
		    ontology_name (str, optional): The name of on ontology.
		"""
		annotations_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			term_ids = row.annotations.split(delim)
			if ontology_name is not None:
				term_ids = [t for t in term_ids if ontology_name.lower() in t.lower()]
			annotations_dict[row.id] = term_ids
		return(annotations_dict)






	def get_description_dictionary(self):
		"""Get a mapping from record IDs to text descriptions.
		
		Returns:
			dict of int:str: Mapping between record IDs and text descriptions.
		"""
		description_dict = {identifier:description for (identifier,description) in zip(self.df.id, self.df.descriptions)}
		return(description_dict)





	def get_species_dictionary(self):
		"""Get a mapping from record IDs to species names.
		
		Returns:
			dict of int:str: Mapping between records IDs and species names.
		"""
		species_dict = {identifier:species for (identifier,species) in zip(self.df.id, self.df.species)}
		return(species_dict)







	def get_ids(self):
		"""Get a list of the IDs for all records in this dataset.
		
		Returns:
			list of int: Unique integer IDs for all the records in this dataset.
		"""
		return(list(self.df.id.values))





	def get_species(self):
		"""Get a list of all the species that are represented in this dataset.
		
		Returns:
			list of str: Names of all the species represented in the dataset.
		"""
		return(list(pd.unique(self.df.species.values)))





	def get_name_to_id_dictionary(self, unambiguous=True):
		"""Get a mapping between gene names or identifiers and the corresponding ID in this dataset.
		
		Args:
			unambiguous (bool, optional): When the unambiguous argument is True then the only 
			keys that are present in the dictionary are names that map to a single gene from the 
			dataset, so this excludes names that map to genes from multiple species. This is 
			because this method is useful for mapping gene names (with no species information) 
			in other files to IDs from this dataset. So those files should only be used when 
			they contain unambiguous names or accessions that encode the species information. 
			When this argument is False this check is not performed and values may be overwritten
			if the same key appears twice across multiple species.
		
		Returns:
			dict of str:int: Mapping between gene names or identifiers and unique integer IDs.
		
		"""


		# TODO create version of this that uses both the species string or number and the gene
		# names as the key to the generated dictionary, so that unambiguous parameter can be 
		# False and genes with the same name from two species can be mapped to the correct IDs.
		name_to_id_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			names = row.unique_gene_identifiers.split(delim)
			for name in names:
				if unambiguous and name not in name_to_id_dict:
					name_to_id_dict[name] = row.id
				else:
					name_to_id_dict[name] = row.id
		return(name_to_id_dict) 


	




	def get_species_to_name_to_ids_dictionary(self, include_synonyms=False, lowercase=False):
		"""Summary
		
		Args:
		    include_synonyms (bool, optional): Description
		    lowercase (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if include_synonyms:
			if lowercase:
				return(self._species_to_name_to_ids_dictionary_with_synonyms_lowercased)
			else:
				return(self._species_to_name_to_ids_dictionary_without_synonyms)
		else:
			if lowercase:
				return(self._species_to_name_to_ids_dictionary_without_synonyms_lowercased)
			else:
				return(self._species_to_name_to_ids_dictionary_without_synonyms)













	######## Merging rows in the dataset based on overlaps in gene names #########


	# The input is the dataframe that has the form:
	# 
	# ID  Names
	# 1	  A
	# 2	  A,B
	# 3	  B,C,D
	# 4   E,F
	# 
	# The behavior of this function should be to add an additional column that has one unique value
	# for all the rows that, based on the assumption that all values in 'Names' are unique identifers,
	# those rows all represent the same actual gene. Note that this might not be a good assumption for 
	# some of the sources of gene synonyms in some of the datasets used, so only values that are likely
	# unique such as true accessions of gene model names should be included in the column used as the 
	# gene names column here. The result of adding a new component column would be as follows.
	# 
	# ID  Names   Component
	# 1	  A        1
	# 2	  A,B      1
	# 3	  B,C,D    1
	# 4   E,F      2
	# 
	# Then the dataframe can be aggregated based on the values in the component column to yield the 
	# following smaller version of the dataframe, with reset ID values to reflect the componets.
	# 
	# ID  Names    Component
	# 1	  A,B,C,D  1
	# 2   E,F      2

	# This is solved by treating the ID values and the name values as unique node names in a graph, 
	# and then given the set of edges that are specifed in that dataframe (1-A, 1-B, ..., 4-F), the
	# problem can be solved as a connected components problem.



	@staticmethod
	def _generate_edges(row, case_sensitive):
		"""
		Given a row in a dataframe that matches the column signature of the one used here, generate
		a list of edges between the entry's ID and each of the gene names mentioned for that entry.
		This uses a representation of the gene names that includes the species as a prefix so that
		the same gene name used for two different species are not considered the same gene. This is 
		treating the gene names and ID's as nodes in a graph for the purpose of collapsing the dataset
		as a connected components problem, see the description in the function to collapse by all 
		gene names below.
		"""
		
		if case_sensitive:
			names = row["unique_gene_identifiers"].split("|")
		else:
			names = row["unique_gene_identifiers"].lower().split("|")

		edges = [(row["id"],"{}.{}".format(row["species"],name)) for name in names]
		return(edges)




	# This is only called by collapse_by_all_gene_names().
	# A method necessary for cleaning up lists of gene identifiers after merging.
	# This removes things from the other gene identifiers if they are already listed as a unique gene identifier.
	# This could happen after merging if some string was unsure about being a unique identifier, but some other entry confirms that is is.
	@staticmethod
	def _remove_duplicate_names(row):
		gene_names = row["unique_gene_identifiers"].split("|")
		gene_synonyms = row["other_gene_identifiers"].split("|")
		updated_gene_synonyms = [x for x in gene_synonyms if x not in gene_names]
		gene_synonyms_str = concatenate_with_delim("|", updated_gene_synonyms)
		return(gene_synonyms_str)





	# This is only called by collapse_by_all_gene_names().
	# Another method necessary for cleaning up lists of gene identifiers after merging.
	# This retains the order except for it puts anything that is also in the gene models column last.
	@staticmethod
	def _reorder_unique_gene_identifers(row):
		unique_identifiers = row["unique_gene_identifiers"].split("|")
		gene_models = row["gene_models"].split("|")
		reordered_unique_identifiers = [x for x in unique_identifiers if x not in gene_models]
		reordered_unique_identifiers.extend(gene_models)
		reordered_unique_identifiers_str = concatenate_with_delim("|", reordered_unique_identifiers)
		return(reordered_unique_identifiers_str)







	def _collapse_by_all_gene_names(self, case_sensitive=False):
		"""Merges all the records where the species and any of the listed gene names or identifiers match. Text descriptions 
		are concatenated and a union of the gene names and ontology term IDs are retained.
		
		Args:
		    case_sensitive (bool, optional): Set to true if gene names that only differ in terms of case
		    should be treated as different genes, by default these genes are considered to be the same gene.
		"""

		# Build the graph model of this data where nodes are gene names or IDs.
		g = nx.Graph()
		edges = self.df.apply(Dataset._generate_edges, case_sensitive=case_sensitive, axis=1)
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
															"unique_gene_identifiers": lambda x: concatenate_with_delim("|",x),
															"other_gene_identifiers": lambda x: concatenate_with_delim("|",x),
															"gene_models": lambda x: concatenate_with_delim("|",x),
															"descriptions":lambda x: concatenate_texts(x),
															"annotations":lambda x: concatenate_with_delim("|",x),
															"sources":lambda x: concatenate_with_delim("|",x)})

		
		# Merging may have resulted in names or identifers being considered by gene names and synonyms.
		# Remove them from the synonym list if they are in the gene name list.
		self.df["other_gene_identifiers"] = self.df.apply(lambda x: Dataset._remove_duplicate_names(x), axis=1)
		self.df["unique_gene_identifiers"] = self.df.apply(lambda x: Dataset._reorder_unique_gene_identifers(x), axis=1)

		# Reset the ID values in the dataset to reflect this change.
		self.df["id"] = None
		self._reset_ids()
		self._update_dictionaries()





		







	######## Summarizing, saving, or converting the whole dataset #########





	def _sort_by_species(self):
		"""Sorts the internal dataframe by the species column.
		"""
		self.df.sort_values(by=["species"], inplace=True)





	def to_csv(self, path):
		"""Writes the dataset to a csv file.
		
		Args:
			path (str): Path of the csv file that will be created.
		"""
		self._sort_by_species()
		self.df.to_csv(path, index=False)








	def to_pandas(self):
		"""Creates a pandas.DataFrame object from this dataset.
		
		Returns:
			pandas.DataFrame: A dataframe representation of this dataset.
		"""
		self._sort_by_species()
		return(self.df)







	def to_json(self):
		"""Creates a nested dictionary from this dataset.
		
		Returns:
		    defaultdict: A nested dictionary representation of this dataset.
		"""
		infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)
		split_on_bar_without_empty_strings = lambda x: [y.strip() for y in x.split("|") if y.strip() != ""]
		json_data = []
		for row in self.df.itertuples():
			d = infinite_defaultdict() 
			d["id"] = row.id
			d["species"] = row.species
			d["unique_gene_identifiers"] = split_on_bar_without_empty_strings(row.unique_gene_identifiers)
			d["other_gene_identifiers"] = split_on_bar_without_empty_strings(row.other_gene_identifiers)
			d["gene_models"] = split_on_bar_without_empty_strings(row.gene_models)
			d["descriptions"] = row.descriptions
			d["annotations"] = split_on_bar_without_empty_strings(row.annotations)
			d["sources"] = split_on_bar_without_empty_strings(row.sources)
			json_data.append(d)
		return(json_data)










	def describe(self):
		"""Returns a summarizing dataframe for this dataset.
		"""

		# Generate a dataframe that summarizes how many genes and unique descriptions there are for each species.
		summary_df = self.df.groupby("species").agg({
			"unique_gene_identifiers": lambda x: len(x), 
			"descriptions": lambda x: len(pd.unique(x))
			})
		summary_df.rename(columns={"gene_names":"num_genes", "descriptions":"unique_descriptions"}, inplace=True)
		summary_df.loc['total']= summary_df.sum()
		summary_df.reset_index(drop=False, inplace=True)
		return(summary_df)































