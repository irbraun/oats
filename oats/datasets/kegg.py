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







#################### Methods specific to using KEGG through the REST API ####################




def get_pathway_dataframe(kegg_species_abbreviation, case_sensitive):
    """
    Create a dictionary mapping KEGG pathways to lists of genes. Code is adapted from the example of
    parsing pathway files obtained through the KEGG REST API, which can be found here:
    https://biopython-tutorial.readthedocs.io/en/latest/notebooks/18%20-%20KEGG.html
    The specifications for those files state that the first 12 characeters of each line are reserved
    for the string which species the section, like "GENE", and the remainder of the line is for 
    everything else.
    
    Args:
        kegg_species_abbreviation (str): Species abbreviation string, see table of options.
        case_sensitive (TYPE): Description
    
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
        pathway_file = REST.kegg_get(dbentries=pathway).read()
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
    if not case_sensitive:
        df["gene_names"] = df["gene_names"].map(str.lower)
        df["ncbi_id"] = df["ncbi_id"].map(str.lower)

    # Update the pathway ID fields using the dictionary.
    df.replace({"pathway_id":pathway_ids_dict}, inplace=True)
    return(df)








def get_pathway_gene_mappings(kegg_species_abbreviation, kegg_pathways_df):
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









def get_id_to_readable_name_mapping(pathways_df):
    df_reduced = pathways_df.drop_duplicates(subset="pathway_id",keep="first", inplace=False)
    id_to_pathway_name = {row.pathway_id:row.pathway_name for row in df_reduced.itertuples()}
    return(id_to_pathway_name)
