from Bio.KEGG import REST
from collections import defaultdict
from glob import glob
import pandas as pd
import numpy as np
import re
import os
import itertools

from oats.nlp.preprocess import concatenate_with_delim
from oats.nlp.small import add_prefix_safely
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
	
	Attributes:
	    df (TYPE): Description
	
	"""
	
	def __init__(self, path, case_sensitive=False):
		"""
		Args:
		    path (str): The path to the CSV file that is used to build this instance. 
		
		    case_sensitive (bool, optional): Set to true to account for differences in capitalization between gene names or identifiers. 
		
		"""

		self.df = pd.read_csv(path)
		self._species_list = self.df["species"].unique()
		self._case_sensitive = case_sensitive
		
		# The forward mapping maps group strings to lists of gene strings.
		# The reverse mapping maps gene strings to lists of group strings.
		# Note that all the gene strings will be lowercase in these dictionaries if _case_sensitive is false.
		# This way, then any new incoming gene strings will also be lowercased before being used as keys in these dictionaries.
		self._species_to_fwd_gene_mappings, self._species_to_rev_gene_mappings = self._get_gene_mappings_from_dataframe(self.df)















	##############  The primary methods that should be used from outside this class  ##############

	def get_groupings_for_dataset(self, dataset):
		"""Returns the 
		
		Args:
		    dataset (oats.datasets.Dataset): A dataset object.
		
		Returns:
		    (dict,dict): A mapping from IDs to lists of group identifiers, and a mapping from group identifiers to lists of IDs.
		"""
		gene_dict = dataset.genes()
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
			return(self._species_to_rev_gene_mappings[species][gene_name.lower()])
		else:
			return(self._species_to_rev_gene_mappings[species][gene_name])



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

























	#################### Methods for using an arbitrary CSV to define groups that genes belong to ####################


	def _get_gene_mappings_from_dataframe(self, df):

		species_to_fwd_dict = defaultdict(lambda: defaultdict(set))
		species_to_rev_dict = defaultdict(lambda: defaultdict(set))

		assert list(df.columns[:3]) == ["species", "group_ids", "gene_identifiers"]
		if not self._case_sensitive:
			df["gene_identifiers"] = df["gene_identifiers"].map(str.lower)
		delim = "|"
		for row in df.itertuples():
			species_code = row.species
			gene_identifiers = row.gene_identifiers.strip().split(delim)
			group_ids = row.group_ids.strip().split(delim)
			for gene_identifier in gene_identifiers:
				for group_id in group_ids:
					species_to_fwd_dict[species_code][group_id].add(gene_identifier)
					species_to_rev_dict[species_code][gene_identifier].add(group_id)

		# Convert the sets of group IDs and gene IDs to lists instead inside the nested dictionaries.
		species_to_fwd_dict = {k1:{k2:list(v2) for k2,v2 in v1.items()} for k1,v1 in species_to_fwd_dict.items()}
		species_to_rev_dict = {k1:{k2:list(v2) for k2,v2 in v1.items()} for k1,v1 in species_to_rev_dict.items()}
		return(species_to_fwd_dict, species_to_rev_dict)
















	#################### Methods specific to using KEGG through the REST API ####################

	@staticmethod
	def save_all_kegg_pathway_files(paths):
		"""Uses the KEGG REST API to find and save all pathway data files for each species in the input dictionary.
		
		Args:
		    paths (dict of str:str): A mapping between strings referencing species and paths to the output directory for each.
		"""
		for species,path in paths.items():
			pathways = REST.kegg_list("pathway", species)
			for pathway in pathways:

				# Get the pathway file contents through the REST API.
				pathway_id = pathway.split()[0]
				pathway_file = REST.kegg_get(dbentries=pathway_id).read()

				# Where should the contents of the obtained file be written?
				pathway_id_str = pathway_id.replace(":","_")
				filename = os.path.join(path, "{}.txt".format(pathway_id_str))
				if not os.path.exists(path):
					os.makedirs(path)
				with open(filename, "w") as outfile:
					outfile.write(pathway_file)






	# This is just a helper method for the larger method below that saves the KEGG pathways files.
	@staticmethod
	def _combine_kegg_gene_identifers(row):
		delim = "|"
		gene_names = row.gene_names.strip().split(delim)
		if not row.ncbi_id == "":
			gene_names.append(add_prefix_safely(row.ncbi_id, NCBI_TAG))
			gene_names.append(row.ncbi_id)
		if not row.uniprot_id == "":
			gene_names.append(add_prefix_safely(row.uniprot_id, UNIPROT_TAG))
		gene_names_str = delim.join(gene_names)
		return(gene_names_str)


	@staticmethod
	def get_dataframe_for_kegg(paths):
		"""
		Create a dictionary mapping KEGG pathways to lists of genes. Code is adapted from the example of
		parsing pathway files obtained through the KEGG REST API, which can be found here:
		https://biopython-tutorial.readthedocs.io/en/latest/notebooks/18%20-%20KEGG.html
		The specifications for those files state that the first 12 characeters of each line are reserved
		for the string which species the section, like "GENE", and the remainder of the line is for 
		everything else.
		
		Args:
		    paths (TYPE): Description
		
		Returns:
		    pandas.DataFrame: The dataframe containing all relevant information about all applicable KEGG pathways.
		
		"""
		dfs_for_each_species = []
		for kegg_species_abbreviation,path in paths.items():

			col_names = ["species", "pathway_id", "pathway_name", "gene_names", "ncbi_id", "uniprot_id", "ko_number", "ec_number"]
			df = pd.DataFrame(columns=col_names)
			pathway_dict_fwd = {}
			pathway_dict_rev = defaultdict(list)
			ko_pathway_ids_dict = {}


			filenames = glob(os.path.join(path,"*.txt"))
			for filename in filenames:
				print(filename)
				with open(filename,"r") as infile:
				

					pathway_file = infile.read()
					for line in pathway_file.rstrip().split("\n"):
						section = line[:12].strip()
						if not section == "":
							current_section = section


						# Note that the pathway ID obtained here is species-specific, and is an identifier for this pathway in this species in KEGG.
						# The actual species-indepenent pathway ID that we want to use is the KO identifier found in the later section.
						if current_section == "ENTRY":
							row_string = line[12:]
							row_tokens = row_string.split()
							pathway_id = row_tokens[0]


						# Should also try and remove the species information from this is using as the pathway name.
						elif current_section == "NAME":
							row_string = line[12:]
							pathway_name = row_string.strip()


						# This isn't currently used for anything, just retaining the information in case we want to look at it later.
						elif current_section == "CLASS":
							row_string = line[12:]
							pathway_class = row_string.strip()


						# Collect information about the gene described on this line.
						elif current_section == "GENE":

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
								"group_ids":"",
								"gene_identifiers":"",
								"pathway_id":pathway_id,
								"pathway_id_for_species":pathway_id,
								"pathway_name":pathway_name,
								"pathway_class":pathway_class,
								"gene_names":names_str,
								"ncbi_id":ncbi_accession,
								"uniprot_id":uniprot_accession,
								"ko_number":ko_accession,
								"ec_number":ec_accession
							}
							df = df.append(row, ignore_index=True, sort=False)

						# Update the dictionary between the species-specific pathway IDs and the species non-speicific KEGG pathways IDs.
						elif current_section == "KO_PATHWAY":
							ko_pathway_id = line[12:].strip()
							ko_pathway_ids_dict[pathway_id] = ko_pathway_id

			# Update the pathway ID field using the dictionary mapping species specific IDs to the general KEGG ones.
			df.replace({"pathway_id":ko_pathway_ids_dict}, inplace=True)
			dfs_for_each_species.append(df)

		# Merge the dataframes that were created for each species, and make sure the columns are organized as expected.
		df = pd.concat(dfs_for_each_species)
		df["group_ids"] = df["pathway_id"]
		df["gene_identifiers"] = df.apply(lambda row: Groupings._combine_kegg_gene_identifers(row), axis=1)
		df = df[["species", "group_ids", "gene_identifiers", "pathway_id", "pathway_id_for_species", "pathway_name", "pathway_class", "gene_names", "ncbi_id", "uniprot_id", "ko_number", "ec_number"]]
		return(df)

		# Using drop duplicates to find the mapping between unique pathway IDs and the longer names for them. Problem is that its 1:many, because the names change based on species unfortunately.
		# TODO address this.
		# def _get_id_to_readable_kegg_name_mapping(self, pathways_df):
		# df_reduced = pathways_df.drop_duplicates(subset="pathway_id",keep="first", inplace=False)
		# id_to_pathway_name = {row.pathway_id:row.pathway_name for row in df_reduced.itertuples()}
		# return(id_to_pathway_name)
































	#################### Methods specific to handling the PlantCyc files obtained through Plant Metabolic Network (PMN) ####################

	@staticmethod
	def get_dataframe_for_plantcyc(paths):


		dfs_for_each_species = []
		for species_code, pathways_filepath in paths.items():

			usecols = ["Pathway-id", "Pathway-name", "Reaction-id", "EC", "Protein-id", "Protein-name", "Gene-id", "Gene-name"]
			usenames = ["pathway_id", "pathway_name", "reaction_id", "ec_number", "protein_id", "protein_name", "gene_id", "gene_name"]
			renamed = {k:v for k,v in zip(usecols,usenames)}
			df = pd.read_table(pathways_filepath, usecols=usecols)
			df.rename(columns=renamed, inplace=True)
			df.fillna("", inplace=True)
			
			# Note, manually reviewed the conventions in gene names for the PlantCyc dataset.
			# The string "unknown" is used for missing values, don't add this as a gene name.
			df.replace(to_replace="unknown", value="", inplace=True)
			combine_columns = lambda row, columns: concatenate_with_delim("|", [row[col] for col in columns])
			df["gene_identifiers"] = df.apply(lambda x: combine_columns(x, ["protein_id", "protein_name", "gene_id", "gene_name"]), axis=1)

			# Some other manipulations to clean up the data, based on how missing value are specified in the PlantCyc Files.
			# Don't retain rows where no gene names are referenced.
			df = df[df["gene_identifiers"]!=""]
			df["ec_number"] = df["ec_number"].map(lambda x: "" if x=="-" else x)
			df["group_ids"] = df["pathway_id"]
			df["species"] = species_code
			df = df[["species", "group_ids", "gene_identifiers", "pathway_id", "pathway_name", "ec_number"]]
			dfs_for_each_species.append(df)

		df = pd.concat(dfs_for_each_species)
		return(df)









	####################  Methods that are useful for interrogating the contents of an instance of this class  ####################


	def to_pandas(self):
		"""Returns that dataframe that this object was constructed with. This dataframe is unchanged from how
		it was read in from the CSV file provided as the main argument. The first three columns are fixed and 
		the remaining columns are unused and could contain any information, they are not removed when the object
		is constructed.

		Returns:
			pandas.DataFrame: The internal dataframe used to define the groupings.
		"""
		return(self.df)





	def describe(self):
		"""Returns a summarizing dataframe for this object.
		
		Returns:
		    pandas.DataFrame: The summarizing dataframe for this object.
		"""
		num_groups_in_each_species = {s:len(self._species_to_fwd_gene_mappings[s].keys()) for s in self._species_list}
		num_genes_mapped_in_each_species = {s:len(self._species_to_rev_gene_mappings[s].keys()) for s in self._species_list}
		summary = pd.DataFrame(self._species_list, columns=["species"])
		summary["num_groups"] = summary["species"].map(num_groups_in_each_species)
		summary["num_genes"] = summary["species"].map(num_genes_mapped_in_each_species)
		return(summary)


















