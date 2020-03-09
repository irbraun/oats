.. oats documentation master file, created by
   sphinx-quickstart on Fri Jan 31 09:37:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``oats``
========

This is a python package created to provide functions for generating and working with similarity networks constructed from natural language descriptions. Specifically, this package was designed for datasets of text descriptions associated with genes, for the purposes of predicting gene function or membership in pathways and regulatory networks through constructing similarity networks with genes and their associated descriptions as nodes. However, many of the functions provided here can generalize to any problem requiring constructing similarity networks based on natural language datasets.


Setup
^^^^^
Run ``pip install dist/oats-0.0.1-py3-none-any.whl`` to install the package with all submodules.




Examples
^^^^^^^^
Add examples here
	
.. code-block:: text

   examples
    




API Reference Summary
^^^^^^^^^^^^^^^^^^^^^

**Annotations** (see :doc:`oats.annotation`)

This subpackage contains classes and methods for reading in ontology files using the OBO format and annotating natural language text using those ontologies. Annotating in this case refers to creating mappings between ontology terms referenced by their unique ontology ID and some instance of text. The ontology class provided in this subpackage is a wrapper for the `pronto <https://pronto.readthedocs.io/en/stable/>`_ ontology class with additional methods for working with sets of terms, calculating distance between terms based on set overlap of information content, and extracting words and vocabularies from term descriptions and synonyms from an ontology. Other functions in this subpackage are for searching for occurences of term mentions in larger strings of text using either the `noble-coder <https://ties.dbmi.pitt.edu/noble-coder/>`_ annotation tool or other searching or string alignment algorithms provided. The main purposes of this subpackage in general is to read in ontologies, use them to produce annotations between text and terms from those ontologies, and find similarity between annotation sets.



**Datasets** (see :doc:`oats.datasets`)

This subpackage contains classes and methods for reading, creating, manipulating, combining, and saving datasets that are based on the relationships between genes and either ontology term annotations or natural language descriptions or both. The dataset class includes methods for creating and adding to such a dataset from csv files, or manipulating and filtering the dataset. Methods for obtaining dictionaries related to this data are provided. The groupings class provides methods for creating or determining relationships between genes from a dataset and any arbitrary group, such as a biochemical pathway, or functional group. These groups can be specified arbitrarily using csv files or can be obtained through existing bioinformatics resources such as `kegg <https://www.genome.jp/kegg/>`_ or `plantcyc <https://www.plantcyc.org/>`_. The main purpose of this subpackage in general is to provided methods for reading in data related to genes, annotations, and text descriptions, and to provied methods for filtering and accessing them as a means of organizing this data for later input into the vectorizing methods or distance matrix methods provided in the distance subpackage.




**Distances** (see :doc:`oats.graphs`)

This subpackage contaings classes and methods for translating strings into numerical vectors, and calculating distance matrices between pairs of text strings that have been translated in this way. The distance module contains methods for handling the embedding by several different approaches, which are wrappers around functions provided by a variety of data science or natural language processing libraries such as `sklearn <https://scikit-learn.org/stable/>`_ and `gensim <https://radimrehurek.com/gensim/>`_. The approaches include using ontology term annotations, n-grams, word embeddings, document embeddings, topic models, and transformer neural networks. For each of these approaches, this subpackage provideds methods for using them to translate text strings to numerical vectors, including control over many parameter choices, and for calculating distance matrices between lists of text strings using these approaches. The main purpose of this subpackage in general is to provide a straight-forward way to obtain distance matrices from collections of text, using a wide range of natural langauge processing techniques so that they can be directly compared.




**NLP** (see :doc:`oats.nlp`)

This subpackage contains functions for working with natural language that were not provided by other packages or are wrapper here to modify behavior or simplify the interface. These include functions such as reducing the vocabulary of an instance of text by accounting for similarity between words, or obtaining words that are related to one another, or determining whether words are domain specific by determining word frequency between domain specific and general text corpora.






API Reference Contents
^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 5

   modules



License
^^^^^^^
.. toctree::
   :maxdepth: 1

   license


Contact
^^^^^^^

This package is a work in progress. Send any feedback, questions, or suggestions to irbraun at iastate dot edu.




Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
