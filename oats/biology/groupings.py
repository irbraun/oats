from Bio.KEGG import REST
from collections import defaultdict
import pandas as pd
import numpy as np
import re
import itertools

from oats.nlp.preprocess import concatenate_with_bar_delim
from oats.nlp.preprocess import add_prefix
from oats.utils.constants import NCBI_TAG, UNIPROT_TAG
from oats.utils.utils import remove_duplicates_retain_order











# The dataframe for each species is only for viewing the input data, should not be used for anything else
# because the columns names are not meant to be consistent between different types of input data, such as 
# searching for pathway information from a database or providing your own CSV file of gene to group mappings.
# Only the number of rows should be allowed to be used as a value to be obtained from these dataframes,
# because it's not dependent on column name and always indicates the number of records looked at generally.

# The pair of dictionaries for each species is always the same however. There is forward mapping between 
# group IDs (no matter what the groups are) and gene names, and a reverse mapping between gene names and 
# group IDs. Those are the structures that should primarily be accessed when using an object of this type,
# however some additional methods are provided as well.

# When providing a CSV, the first two columns are always used, and no header should be included. The first
# column is assumed to be group IDs (bar delimited if there are more than one mentioned on a line), and the
# second column is assumed to be gene names (bar delimited if there are more than one mentioned on a line).
# For a given row, all gene names are assumed to be members of all group IDs mentioned.

# If case sensitive is false, then the internal dictionaries will only hold lowercase versions of all gene
# names, and when generating dictionaries from outside lists of gene names, they will be converted to lower
# case before any matching to the internal gene names is done.




class Groupings:

	"""
	This is a class for creating and containing mappings between information in the dataset of interest
	and relationships to other types of information such as biochemical pathways or protein-protein 
	interactions.

	"""
	
	def __init__(self, species_dict, source, name_mapping=None, case_sensitive=False):
		"""
		Args:
			species_dict (dict of str:str): A mapping where keys are species names and values are file paths with data for that species.
			Not every type of grouping object that can be created needs files to be specified in this dictionary. The values can be 
			missing if KEGG is used for example, because the KEGG REST API is used to obtain the information related to the pathways.

			source (str): Indicates where the information for the groups is pulled from. Can be "kegg", "pmn", "plantcyc", or "csv". If "kegg"
			is used then the values in species_dict argument are not used, otherwise they are. If "pmn" or "plantcyc" is used the files 
			should be the PlantCyc files for that particular species. If the "csv" is used the files should be contain a mapping between gene
			names or identifiers in the first column and strings that represent groups in the second column. Multiple entries can be present
			in either column by separating with a pipe. All gene listed on given line are considered mapped to all groups listed on a given 
			line (because genes can belong to more than one group).
			
			name_mapping (dict of obj:str, optional): This argument is only necessary if specifying groups using csv files. It should be a 
			dictionary mapping integers or strings (whatever the datatype of the group IDs are) to strings that represent the full name of those
			groups, which are useful for outputtting to tables. 
			
			case_sensitive (bool, optional): Set to true to account for differences in capitalization between gene names or identifiers in
			the dataset used and in the databases or files used to create these groups.
		
		Raises:
			ValueError: The name of the source to create the groupings from is not recognized or supported.
		"""
		# Create one dataframe and pair of dictionaries for each species code.
		self._species_list = list(species_dict.keys())
		self._species_to_df_dict = {}
		self._species_to_fwd_gene_mappings = {}
		self._species_to_rev_gene_mappings = {}
		self._readable_name_mappings = {}
		self._case_sensitive = case_sensitive
		for species,path in species_dict.items():
			

			# Use the PlantCyc files from PMN as the source of the groupings.
			if source.lower() == "plantcyc" or source.lower() == "pmn":
				df = self._get_dataframe_from_pmn(species, path, self._case_sensitive)
				fwd_mapping, rev_mapping = self._get_gene_mappings_from_pmn(species, df)
				self._readable_name_mappings.update(self._get_id_to_readable_pmn_name_mapping(df))
			

			# Use the KEGG REST API to obtain the groupings.
			elif source.lower() == "kegg":
				df = self._get_dataframe_from_kegg(species, self._case_sensitive)
				fwd_mapping, rev_mapping = self._get_gene_mappings_from_kegg(species, df)
				self._readable_name_mappings.update(self._get_id_to_readable_kegg_name_mapping(df))


			# Use any arbitrary CSV file that is provided as the soure for the groupings.
			elif source.lower() == "csv":
				df = self._get_dataframe_from_csv(species,path)
				fwd_mapping, rev_mapping = self._get_gene_mappings_from_csv(species, df)
				self._readable_name_mappings.update(name_mapping)


			# Not supporting whatever type of source string was provided yet.
			else:
				raise ValueError("name of groupings source ({}) not recognized".format(source))

			self._species_to_df_dict[species] = df            
			self._species_to_fwd_gene_mappings[species] = fwd_mapping
			self._species_to_rev_gene_mappings[species] = rev_mapping















	##############  The primary methods that should be used from outside this class  ##############

	def get_groupings_for_dataset(self, dataset):
		"""Returns the 
		
		Args:
		    dataset (oats.datasets.Dataset): A dataset object.
		
		Returns:
		    (dict,dict): A mapping from IDs to lists of group identifiers, and a mapping from group identifiers to lists of IDs.
		"""
		gene_dict = dataset.get_gene_dictionary()
		id_to_group_ids_dict = self.get_id_to_group_ids_dict(gene_dict)
		group_id_to_ids_dict = self.get_group_id_to_ids_dict(gene_dict)
		return(id_to_group_ids_dict, group_id_to_ids_dict)









	def get_id_to_group_ids_dict(self, gene_dict):
		"""Returns a mapping from IDs to lists of group IDs. Note that this retains as keys even IDs that don't map to any groups.
		
		Args:
			gene_dict (dict of int:oats.datasets.Gene): Mapping between unique integer IDs from the dataset and the corresponding gene objects. 
		
		Returns:
			dict of int:list of obj: Mapping between unique integer IDs from the dataset and list of group IDs. 
		"""
		membership_dict = {}
		for gene_id, gene_obj in gene_dict.items():
			membership_dict[gene_id] = self.get_group_ids_from_gene_obj(gene_obj)
		membership_dict = {k:remove_duplicates_retain_order(v) for k,v in membership_dict.items()}
		return(membership_dict)





	def get_group_id_to_ids_dict(self, gene_dict):
		"""Returns a mapping from group IDs to lists of IDs. Note that groups are only retained as keys if they are mapped to atleast one ID.
		
		Args:
			gene_dict (dict of int:oats.dataset.Gene): Mapping between unique integer IDs from the dataset and the corresponding gene objects. 
		
		Returns:
			dict of obj:list of int: Mapping between integers or strings (whatever datatype the group IDs are given as) and lists of unique integer IDs from the dataset.
		"""
		reverse_membership_dict = defaultdict(list)
		for gene_id, gene_obj in gene_dict.items():
			group_ids = self.get_group_ids_from_gene_obj(gene_obj)
			for group_id in group_ids:
				reverse_membership_dict[group_id].append(gene_id)
		reverse_membership_dict = {k:remove_duplicates_retain_order(v) for k,v in reverse_membership_dict.items()}      
		return(reverse_membership_dict)





	def get_long_name(self, group_id):
		"""Returns a string which is the long and readable name for this group ID.
		
		Args:
			group_id (obj): Any group ID.
		
		Returns:
			str: The full name for this group given as a string.
		"""
		return(self._readable_name_mappings[group_id])











	##############  Some secondary methods that might also be useful from outside this class  ##############



	def get_group_ids_from_gene_obj(self, gene_obj):
		"""Given a gene object, return a list of group IDs it belongs in.
		
		Args:
			gene_obj (TYPE): Description
		
		Returns:
			TYPE: Description
		"""
		group_ids = []
		species = gene_obj.species
		gene_names = gene_obj.unique_identifiers
		if not self._case_sensitive:
			group_ids.extend(list(itertools.chain.from_iterable([self._species_to_rev_gene_mappings.get(species,{}).get(name.lower(),[]) for name in gene_names])))
		else:
			group_ids.extend(list(itertools.chain.from_iterable([self._species_to_rev_gene_mappings.get(species,{}).get(name,[]) for name in gene_names])))
		return(group_ids)



	def get_group_ids_from_gene_name(self, species, gene_name):
		"""
		Given a species code (three letters) and a gene name, return a list of group IDs that
		the gene might belong to.
		
		Args:
			species (TYPE): Description
			gene_name (TYPE): Description
		
		Returns:
			TYPE: Description
		"""
		if not self._case_sensitive:
			return(self._species_to_rev_gene_mappings[species][name.lower()])
		else:
			return(self._species_to_rev_gene_mappings[species][name])



	def get_gene_names_from_group_id(self, species, group_id):
		"""
		Given a group ID and species code (three letters), return a list of all the gene names 
		which are associated with that ID in this instance of this class.
		
		Args:
			species (TYPE): Description
			
			group_id (TYPE): Description
		
		Returns:
			TYPE: Description
		"""
		return(self._species_to_fwd_gene_mappings[species][group_id])

























	#################### Methods for using an arbitrary CSV to define pathways (or pathway-like groups) ####################

	def _get_dataframe_from_csv(self, species_code, filepath):
		df = pd.read_csv(filepath,usecols=[0,1])
		df.columns = ["group_id","gene_names"]
		df["species"] = species_code
		df = df[["species","group_id","gene_names"]]
		if not self._case_sensitive:
			df["gene_names"] = df["gene_names"].map(str.lower)
		return(df)

	def _get_gene_mappings_from_csv(self, species_code, df):
		group_dict_fwd = defaultdict(list)
		group_dict_rev = defaultdict(list)
		delim = "|"
		for row in df.itertuples():
			gene_names = row.gene_names.strip().split(delim)
			group_ids = row.group_id.strip().split(delim)
			for gene_name in gene_names:
				for group_id in group_ids:
					group_dict_fwd[group_id].append(gene_name)
					group_dict_rev[gene_name].append(group_id)
		return(group_dict_fwd, group_dict_rev)













	#################### Methods specific to using KEGG through the REST API ####################

	def _get_dataframe_from_kegg(self, kegg_species_abbreviation, _case_sensitive):
		"""
		Create a dictionary mapping KEGG pathways to lists of genes. Code is adapted from the example of
		parsing pathway files obtained through the KEGG REST API, which can be found here:
		https://biopython-tutorial.readthedocs.io/en/latest/notebooks/18%20-%20KEGG.html
		The specifications for those files state that the first 12 characeters of each line are reserved
		for the string which species the section, like "GENE", and the remainder of the line is for 
		everything else.
		
		Args:
			kegg_species_abbreviation (str): Species abbreviation string, see table of options.
			_case_sensitive (TYPE): Description
		
		Returns:
			pandas.DataFrame: The dataframe containing all relevant information about all applicable KEGG pathways.
		"""

		col_names = ["species", "pathway_id", "pathway_name", "gene_names", "ncbi_id", "uniprot_id", "ko_number", "ec_number"]
		df = pd.DataFrame(columns=col_names)

		pathway_dict_fwd = {}
		pathway_dict_rev = defaultdict(list)
		pathways = REST.kegg_list("pathway", kegg_species_abbreviation)
		pathway_ids_dict = {}

		for pathway in pathways:

			pathway_id = pathway.split()[0]
			pathway_file = REST.kegg_get(dbentries=pathway_id).read()
			


			for line in pathway_file.rstrip().split("\n"):
				section = line[:12].strip()
				if not section == "":
					current_section = section

				# Collect information about the gene described on this line.
				if current_section == "GENE":

					# Parse this line of the pathway file.
					row_string = line[12:]
					row_tokens = line[12:].split()
					ncbi_accession = row_tokens[0]
					uniprot_accession = ""

					# Handing the gene names and other accessions with regex.
					names_portion_without_accessions = " ".join(row_tokens[1:])
					pattern_for_ko = r"(\[[A-Za-z0-9_|\.|:]*?KO[A-Za-z0-9_|\.|:]*?\])"
					pattern_for_ec = r"(\[[A-Za-z0-9_|\.|:]*?EC[A-Za-z0-9_|\.|:]*?\])"
					result_for_ko = re.search(pattern_for_ko, row_string)
					result_for_ec = re.search(pattern_for_ec, row_string)
					if result_for_ko == None:
						ko_accession = ""
					else:
						ko_accession = result_for_ko.group(1)
						names_portion_without_accessions = names_portion_without_accessions.replace(ko_accession, "")
						ko_accession = ko_accession[1:-1]
					if result_for_ec == None:
						ec_accession = ""
					else:
						ec_accession = result_for_ec.group(1)
						names_portion_without_accessions = names_portion_without_accessions.replace(ec_accession, "")
						ec_accession = ec_accession[1:-1]

					# Parse the other different names or symbols mentioned.
					names = names_portion_without_accessions.split(";")
					names = [name.strip() for name in names]
					names_delim = "|"
					names_str = names_delim.join(names)


					# Update the dataframe no matter what the species was.
					row = {
						"species":kegg_species_abbreviation,
						"pathway_id":pathway,
						"pathway_name":pathway,
						"gene_names":names_str,
						"ncbi_id":ncbi_accession,
						"uniprot_id":uniprot_accession,
						"ko_number":ko_accession,
						"ec_number":ec_accession
					}
					df = df.append(row, ignore_index=True, sort=False)

				# Update the dictionary between pathway names and IDs.
				if current_section == "KO_PATHWAY":
					pathway_id = line[12:].strip()
					pathway_ids_dict[pathway] = pathway_id

		# Convert the gene names and identifiers to lowercase if case sensitive not set.
		if not _case_sensitive:
			df["gene_names"] = df["gene_names"].map(str.lower)
			df["ncbi_id"] = df["ncbi_id"].map(str.lower)

		# Update the pathway ID fields using the dictionary.
		df.replace({"pathway_id":pathway_ids_dict}, inplace=True)
		return(df)


	def _get_gene_mappings_from_kegg(self, kegg_species_abbreviation, kegg_pathways_df):
		""" Obtain forward and reverse mappings between pathways and gene names.
		Args:
			kegg_species_abbreviation (str): The species code for which genes to look at.
			kegg_pathways_df (pandas.DataFrame): The dataframe containing all the pathway information.
		Returns:
			(dict,dict): A mapping from pathway IDs to lists of gene names,
						 and a mapping from gene names to lists of pathway IDs. 
		"""
		pathway_dict_fwd = defaultdict(list)
		pathway_dict_rev = defaultdict(list)
		delim = "|"
		for row in kegg_pathways_df.itertuples():
			gene_names = row.gene_names.strip().split(delim)
			if not row.ncbi_id == "":
				gene_names.append(add_prefix(row.ncbi_id, NCBI_TAG))
				gene_names.append(row.ncbi_id)
			if not row.uniprot_id == "":
				gene_names.append(add_prefix(row.uniprot_id, UNIPROT_TAG))
			for gene_name in gene_names:
				pathway_dict_fwd[row.pathway_id].append(gene_name)
				pathway_dict_rev[gene_name].append(row.pathway_id)
		return(pathway_dict_fwd, pathway_dict_rev)


	def _get_id_to_readable_kegg_name_mapping(self, pathways_df):
		df_reduced = pathways_df.drop_duplicates(subset="pathway_id",keep="first", inplace=False)
		id_to_pathway_name = {row.pathway_id:row.pathway_name for row in df_reduced.itertuples()}
		return(id_to_pathway_name)
































	#################### Methods specific to handling the PlantCyc files obtained through Plant Metabolic Network (PMN) ####################


	def _get_dataframe_from_pmn(self, species_code, pathways_filepath, _case_sensitive):
		usecols = ["Pathway-id", "Pathway-name", "Reaction-id", "EC", "Protein-id", "Protein-name", "Gene-id", "Gene-name"]
		usenames = ["pathway_id", "pathway_name", "reaction_id", "ec_number", "protein_id", "protein_name", "gene_id", "gene_name"]
		renamed = {k:v for k,v in zip(usecols,usenames)}
		df = pd.read_table(pathways_filepath, usecols=usecols)
		df.rename(columns=renamed, inplace=True)
		df.fillna("", inplace=True)
		
		# Note, manually reviewed the conventions in gene names for the PlantCyc dataset.
		# The string "unknown" is used for missing values, don't add this as a gene name.
		df.replace(to_replace="unknown", value="", inplace=True)

		df["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df["protein_id"], df["protein_name"], df["gene_id"], df["gene_name"])
		if not _case_sensitive:
			df["gene_names"] = df["gene_names"].map(str.lower)
		df["species"] = species_code
		df = df[["species", "pathway_id", "pathway_name", "gene_names", "ec_number"]]
		return(df)



	def _get_gene_mappings_from_pmn(self, species_code, pathways_df):
		pathway_dict_fwd = defaultdict(list)
		pathway_dict_rev = defaultdict(list)
		delim = "|"
		for row in pathways_df.itertuples():
			gene_names = row.gene_names.strip().split(delim)
			for gene_name in gene_names:
				pathway_dict_fwd[row.pathway_id].append(gene_name)
				pathway_dict_rev[gene_name].append(row.pathway_id)
		return(pathway_dict_fwd, pathway_dict_rev)





	def _get_id_to_readable_pmn_name_mapping(self, pathways_df):
		df_reduced = pathways_df.drop_duplicates(subset="pathway_id",keep="first", inplace=False)
		id_to_pathway_name = {row.pathway_id:row.pathway_name for row in df_reduced.itertuples()}
		return(id_to_pathway_name)



















































	####################  Methods that are useful for interrogating the contents of an instance of this class  ####################


	def to_csv(self, path):
		"""Write the contents of internal dataframe for this object to a file.
		
		Args:
			path (str): Path to the csv file that will be created.
		"""
		df = pd.concat(self._species_to_df_dict.values(), ignore_index=True) 
		df.to_csv(path, index=False)

	def to_pandas(self):
		"""Get the internal dataframe used for this grouping object. The columns of the 
		dataframe can vary based on what the groups for this object are and how they were 
		found.

		Returns:
			pandas.DataFrame: The internal dataframe used for this grouping.
		"""
		df = pd.concat(self._species_to_df_dict.values(), ignore_index=True) 
		return(df)



	def describe(self):
		"""Prints a quick summary of this objects contents.
		"""
		print("Number of groups present for each species")
		for species in self._species_list:
			print("  {}: {}".format(species, len(self._species_to_fwd_gene_mappings[species].keys())))
		print("Number of genes names mapped to any group for each species")
		for species in self._species_list:
			print("  {}: {}".format(species, len(self._species_to_rev_gene_mappings[species].keys())))




















