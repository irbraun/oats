from collections import defaultdict
import pandas as pd
import numpy as np
import re
import itertools

from oats.nlp.preprocess import concatenate_with_bar_delim
from oats.nlp.preprocess import add_prefix
from oats.utils.constants import NCBI_TAG, UNIPROT_TAG
from oats.utils.utils import remove_duplicates_retain_order







#################### Methods specific to handling the PlantCyc files obtained through PMN ####################





def get_pathway_dataframe(species_code, pathways_filepath, case_sensitive):
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
    if not case_sensitive:
        df["gene_names"] = df["gene_names"].map(str.lower)
    df["species"] = species_code
    df = df[["species", "pathway_id", "pathway_name", "gene_names", "ec_number"]]
    return(df)






def get_pathway_gene_mappings(species_code, pathways_df):
    pathway_dict_fwd = defaultdict(list)
    pathway_dict_rev = defaultdict(list)
    delim = "|"
    for row in pathways_df.itertuples():
        gene_names = row.gene_names.strip().split(delim)
        for gene_name in gene_names:
            pathway_dict_fwd[row.pathway_id].append(gene_name)
            pathway_dict_rev[gene_name].append(row.pathway_id)
    return(pathway_dict_fwd, pathway_dict_rev)







def get_id_to_readable_name_mapping(pathways_df):
    df_reduced = pathways_df.drop_duplicates(subset="pathway_id",keep="first", inplace=False)
    id_to_pathway_name = {row.pathway_id:row.pathway_name for row in df_reduced.itertuples()}
    return(id_to_pathway_name)

