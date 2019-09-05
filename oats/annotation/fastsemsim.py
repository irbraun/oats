from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import os
import sys
import glob
import math
import re

from oats.nlp.search import binary_search_rabin_karp










def check_ontology(ontology_source_file):

	# Parameters for the ontology file.
	ontology_file_type = "obo"
	ontology_type = "Ontology"
	ignore_parameters = {}

	print("\n######################")
	print("# Loading ontology... #")
	print("######################\n")

	# Load the file.
	ontology = fss.load_ontology(source_file=ontology_source_file, ontology_type=ontology_type, file_type=ontology_file_type)

	print("\n#################################")
	print("# Ontology successfully loaded.")
	print("#################################\n")

	print("source_file: " + str(ontology_source_file))
	print("file_type: " + str(ontology_file_type))
	print("ontology_type: " + str(ontology_type))
	print("ignore_parameters: " + str(ignore_parameters))
	print("Number of nodes: " + str(ontology.node_number()))
	print("Number of edges: " + str(ontology.edge_number()))
	print("\nRoot terms in the ontology:\n-------------\n" + str(ontology.roots))
	print("\nType and number of edges:\n-------------\n" + str(ontology.edges['type'].value_counts()))
	print("-------------")
	print("\nInner edge number (within the ontology):\n-------------\n" + str(ontology.edges['inner'].value_counts()))
	print("-------------")
	print("\nIntra edge number (within the same namespace):\n-------------\n" + str(ontology.edges['intra'].value_counts()))
	print("-------------")
	print("\nOuter edges (link to other ontologies):\n-------------\n" + str(ontology.edges.loc[ontology.edges['inner'] == False]))
	print("-------------")
	print("\nInter edges (link between different namespaces - within the same ontology):\n-------------\n" + str(ontology.edges.loc[(ontology.edges['intra'] == False) & (ontology.edges['inner'] == True)]))
	print("-------------")















def check_annotations(ac_source_file):

	# Parameters for annotation corpus file with descriptions from fastsemsim documentation.
	ac_source_file_type = "plain"
	ac_params = {}
	ac_params['multiple'] = True 	# Set to True if there are many associations per line (the object in the first field is associated to all the objects in the other fields within the same line).
	ac_params['term first'] = False # Set to True if the first field of each row is a GO term. Set to False if the first field represents a protein/gene.
	ac_params['separator'] = "\t" 	# Select the separtor used to divide fields.

	print("\n#################################")
	print("# Loading annotation corpus.")
	print("#################################\n")

	# Load the file.
	ac = fss.load_ac(ontology, source_file=ac_source_file, file_type=ac_source_file_type, species=None, ac_descriptor=None, params = ac_params)

	print("\n#################################")
	print("# Annotation corpus successfully loaded.")
	print("#################################\n")

	print("\n\n")
	print("AC source: " + str(ac_source_file))
	print("ac source_type: " + str(ac_source_file_type))
	print("ac_parameters: " + str(ac_params))
	print("ac - Number of annotated proteins: " + str(len(ac.annotations)))
	print("ac - Number of annotated terms: " + str(len(ac.reverse_annotations)))
	print("The set of objects is: ", ac.obj_set)
	print("The set of terms is: ", ac.term_set)
	print("-------------")


