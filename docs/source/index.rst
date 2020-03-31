.. oats documentation master file, created by
   sphinx-quickstart on Fri Jan 31 09:37:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``oats``
========

``oats`` is a python package created to provide functions for evaluting similarity between text using a wide variety of approaches including natural language processing, machine learning, and semantic annotation using ontologies. This package was specifically created to be used in a project to assess similarity between phenotype descriptions across plant species in order to better organize this data and make biological predictions. For this reason, some of the subpackages are specifically oriented towards working with biological datasets, although many of the functions are applicable outside of this domain. In general, the purpose of this package is to provide useful wrappers or shortcut functions on top of a number of existing packages and tools for working with biological datasets, ontologies, and natural language in order to make analysis of this kind of data easier. 


Setup
^^^^^

Use pip to install with

.. code-block:: text

   $ pip install dist/oats-0.0.1-py3-none-any.whl






API Reference Summary
^^^^^^^^^^^^^^^^^^^^^

**Ontologies and Annotations** (see :doc:`oats.annotation`)

This subpackage contains classes and methods for reading in ontology files using the OBO format and annotating datasets of text using terms from those ontologies. Annotating in this case refers to creating mappings between ontology terms and some instance of text. The class used for ontologies is a wrapper for the ontology class from `pronto <https://pronto.readthedocs.io/en/stable/>`_ with some additional methods for working with ontologies in the context of semantic annotation and natural language processing problems. This subpackage provides functions for mapping ontology terms to text using existing tools like the `NOBLE Coder <https://ties.dbmi.pitt.edu/noble-coder/>`_ semantic annotation tool and other text alignment algorithms, and extracting the dictionaries that contain the resulting annotations. The primary purposes of this subpackage are to read in ontologies and use them to produce annotations between text and terms.


.. toctree::
   :maxdepth: 5

   oats.annotation







**Biological Datasets** (see :doc:`oats.biology`)

This subpackage contains classes and methods for reading, creating, manipulating, combining, and saving datasets that are based on the relationships between genes and either ontology term annotations or natural language descriptions or both. The dataset class includes methods for creating and adding to such a dataset from csv files, or manipulating and filtering the dataset. Methods for obtaining dictionaries related to this data are provided. The groupings class provides methods for creating or determining relationships between genes from a dataset and any arbitrary group, such as a biochemical pathway, or functional group. These groups can be specified arbitrarily using csv files or can be obtained through existing bioinformatics resources such as `KEGG <https://www.genome.jp/kegg/>`_ or `PlantCyc <https://www.plantcyc.org/>`_. The main purpose of this subpackage in general is to provided methods for reading in data related to genes, annotations, and text descriptions, and to provide methods for filtering and accessing them as a means of organizing this data for later input into the vectorizing methods or distance matrix methods provided in the other subpackages.

.. toctree::
   :maxdepth: 5

   oats.biology





**Text Distances** (see :doc:`oats.distances`)

This subpackage contaings classes and methods for translating strings into numerical vectors, and calculating distance matrices between pairs of text strings that have been translated in this way. The distance module contains methods for handling the embedding by several different approaches, which are wrappers around functions provided by a variety of data science, natural language processing, and machine learning libraries such as `sklearn <https://scikit-learn.org/stable/>`_ and `gensim <https://radimrehurek.com/gensim/>`_. The approaches include using ontology term annotations, n-grams, word embeddings, document embeddings, topic models, and transformer neural networks. For each of these approaches, this subpackage provideds methods for using them to translate text strings to numerical vectors, including control over many parameter choices, and for calculating distance matrices between lists of text strings using these approaches. The main purpose of this subpackage in general is to provide a straight-forward way to obtain distance matrices from collections of text, using a wide range of natural langauge processing techniques so that they can be directly compared.

.. toctree::
   :maxdepth: 5

   oats.distances




**NLP** (see :doc:`oats.nlp`)

This subpackage contains functions for working with natural language that were either not provided by other packages or are wrapped here to modify behavior or simplify the interface. These include functions such as reducing the vocabulary of an instance of text by accounting for similarity between words, or obtaining words that are related to one another, or determining whether words are domain specific by determining word frequency between domain specific and general text corpora.

.. toctree::
   :maxdepth: 5

   oats.nlp






API Reference Contents
^^^^^^^^^^^^^^^^^^^^^^
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

Send any feedback, questions, or suggestions to irbraun at iastate dot edu.




Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
