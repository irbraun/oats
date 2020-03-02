# OATS: Ontology Annotation and Text Similarity

This is a python package created to provide functions for generating and working with networks constructed from natural language descriptions. Specifically, this package was designed for datasets of text descriptions associated with genes, for the purposes of predicting gene function or membership in pathways and regulatory networks through constructing similarity networks with genes and their associated descriptions as nodes. However, many of the functions provided here can generalize to any problem requiring constructing similarity networks based on natural language datasets.

#### Functions specific to biological datasets
1. Preprocessing, cleaning, and organizing datasets of gene names and accessions.
2. Obtaining and organizing data related to biological pathways and protein-protein interactions.
3. Integrating that information from KEGG, Plant Metabolic Network, and STRING with user datasets.

#### Functions with general uses
1. Preprocessing, cleaning, and organizing text descriptions of any length.
2. Annotating datasets of text descriptions with terms from an input ontology.
3. Applying NLP methods such as bag-of-words and document embedding to datasets of text descriptions.
4. Constructing similarity networks for text descriptions using the above methods.
5. Training and testing machine learning models for combining methods to join networks.

### Feedback
This package is a work in progress. Send any feedback, questions, or suggestions to irbraun at iastate dot edu.