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
from oats.datasets import pmn
from oats.datasets import kegg





class Groupings:




    def __init__(self, species_dict, source, name_mapping=None, case_sensitive=False):



        """
        The dataframe for each species is only for viewing the input data, should not be used for anything else
        because the columns names are not meant to be consistent between different types of input data, such as 
        searching for pathway information from a database or providing your own CSV file of gene to group mappings.
        Only the number of rows should be allowed to be used as a value to be obtained from these dataframes,
        because it's not dependent on column name and always indicates the number of records looked at generally.

        The pair of dictionaries for each species is always the same however. There is forward mapping between 
        group IDs (no matter what the groups are) and gene names, and a reverse mapping between gene names and 
        group IDs. Those are the structures that should primarily be accessed when using an object of this type,
        however some additional methods are provided as well.

        When providing a CSV, the first two columns are always used, and no header should be included. The first
        column is assumed to be group IDs (bar delimited if there are more than one mentioned on a line), and the
        second column is assumed to be gene names (bar delimited if there are more than one mentioned on a line).
        For a given row, all gene names are assumed to be members of all group IDs mentioned.

        If case sensitive is false, then the internal dictionaries will only hold lowercase versions of all gene
        names, and when generating dictionaries from outside lists of gene names, they will be converted to lower
        case before any matching to the internal gene names is done.
        """


        # Create one dataframe and pair of dictionaries for each species code.
        self.species_list = list(species_dict.keys())
        self.species_to_df_dict = {}
        self.species_to_fwd_gene_mappings = {}
        self.species_to_rev_gene_mappings = {}
        self.readable_name_mappings = {}
        self.case_sensitive = case_sensitive
        for species,path in species_dict.items():
            
            # Use the PlantCyc files from PMN as the source of the groupings.
            if source.lower() == "plantcyc" or source.lower() == "pmn":
                df = pmn.get_pathway_dataframe(species,path, case_sensitive)
                fwd_mapping, rev_mapping = pmn.get_pathway_gene_mappings(species, df)
                self.readable_name_mappings.update(pmn.get_id_to_readable_name_mapping(df))
            
            # Use the KEGG REST API to obtain the groupings.
            elif source.lower() == "kegg":
                df = kegg.get_pathway_dataframe(species, case_sensitive)
                fwd_mapping, rev_mapping = kegg.get_pathway_gene_mappings(species, df)
                self.readable_name_mappings.update(kegg.get_id_to_readable_name_mapping(df))

            # Use any arbitrary CSV file that is provided as the soure for the groupings.
            elif source.lower() == "csv":
                df = self._get_dataframe_from_csv(species,path)
                fwd_mapping, rev_mapping = self._get_gene_mappings_from_csv(species, df)
                self.readable_name_mappings.update(name_mapping)

            # Not supporting whatever type of source string was provided yet.
            else:
                raise ValueError("name of groupings source ({}) not recognized".format(source))

            self.species_to_df_dict[species] = df            
            self.species_to_fwd_gene_mappings[species] = fwd_mapping
            self.species_to_rev_gene_mappings[species] = rev_mapping















    ##############  The primary methods that should be used from outside this class  ##############



    def get_id_to_group_ids_dict(self, gene_dict):
        """ Returns a mapping from object IDs to lists of group IDs. Note that this retains as keys even IDs that don't map to any groups.
        Args:
            gene_dict (TYPE): Description
        Returns:
            TYPE: Description
        """
        membership_dict = {}
        for gene_id, gene_obj in gene_dict.items():
            membership_dict[gene_id] = self.get_group_ids_from_gene_obj(gene_obj)
        membership_dict = {k:remove_duplicates_retain_order(v) for k,v in membership_dict.items()}
        return(membership_dict)


    def get_group_id_to_ids_dict(self, gene_dict):
        """Returns a mapping from group IDs to lists of object IDs. Note that groups are only keys if they are mapped to atleast one ID.
        Args:
            gene_dict (TYPE): Description
        Returns:
            TYPE: Description
        """
        reverse_membership_dict = defaultdict(list)
        for gene_id, gene_obj in gene_dict.items():
            group_ids = self.get_group_ids_from_gene_obj(gene_obj)
            for group_id in group_ids:
                reverse_membership_dict[group_id].append(gene_id)
        reverse_membership_dict = {k:remove_duplicates_retain_order(v) for k,v in reverse_membership_dict.items()}      
        return(reverse_membership_dict)



    def get_long_name(self, group_id):
        """Returns a string which is the long and readable namea for this group ID.
        Args:
            group_id (TYPE): Description
        Returns:
            TYPE: Description
        """
        return(self.readable_name_mappings[group_id])











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
        gene_names = gene_obj.names
        if not self.case_sensitive:
            group_ids.extend(list(itertools.chain.from_iterable([self.species_to_rev_gene_mappings.get(species,{}).get(name.lower(),[]) for name in gene_names])))
        else:
            group_ids.extend(list(itertools.chain.from_iterable([self.species_to_rev_gene_mappings.get(species,{}).get(name,[]) for name in gene_names])))
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
        if not self.case_sensitive:
            return(self.species_to_rev_gene_mappings[species][name.lower()])
        else:
            return(self.species_to_rev_gene_mappings[species][name])



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
        return(self.species_to_fwd_gene_mappings[species][group_id])































    #################### Methods for using an arbitrary CSV to define pathways (or pathway-like groups) ####################

    def _get_dataframe_from_csv(self, species_code, filepath):
        df = pd.read_csv(filepath,usecols=[0,1])
        df.columns = ["group_id","gene_names"]
        df["species"] = species_code
        df = df[["species","group_id","gene_names"]]
        if not self.case_sensitive:
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


















    ####################  Methods that are useful for interrogating the contents of an instance of this class  ####################





    def to_csv(self, path):
        """
        Write the contents of the combined dataframe for all the pathway inforomation
        out to a csv file. 
        Args:
            path (str): Full path to where the output file should be written.
        """
        df = pd.concat(self.species_to_df_dict.values(), ignore_index=True) 
        df.to_csv(path, index=False)


    def to_pandas(self):
        """
        Returns a df with the contents of the combined dataframe for all the pathway
        information that is used for this dataset. This the same dataframe that is
        built when the function to write the objecj to a csv is called.
        Returns:
            pandas.DataFrame: The resulting dataframe object.
        """
        df = pd.concat(self.species_to_df_dict.values(), ignore_index=True) 
        return(df)


    def describe(self):
        """
        Write out information about what is contained within this groupings object.
        """
        print("Number of groups present for each species")
        for species in self.species_list:
            print("  {}: {}".format(species, len(self.species_to_fwd_gene_mappings[species].keys())))
        print("Number of genes names mapped to any group for each species")
        for species in self.species_list:
            print("  {}: {}".format(species, len(self.species_to_rev_gene_mappings[species].keys())))




















