.. oats documentation master file, created by
   sphinx-quickstart on Fri Jan 31 09:37:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``oats``
========

This is a python package created to provide functions for generating and working with networks constructed from natural language descriptions. Specifically, this package was designed for datasets of text descriptions associated with genes, for the purposes of predicting gene function or membership in pathways and regulatory networks through constructing similarity networks with genes and their associated descriptions as nodes. However, many of the functions provided here can generalize to any problem requiring constructing similarity networks based on natural language datasets.


Setup
^^^^^
Run ``pip install dist/oats-0.0.1-py3-none-any.whl`` to install the package and all submodules.




Examples
^^^^^^^^
Some examples go here:..
	
.. code-block:: text

   pip install dist/oats-0.0.1-py3-none-any.whl
    
after the code



API Reference Summary
^^^^^^^^^^^^^^^^^^^^^

**Annotations** (see :doc:`oats.annotation`)

Contains classes and methods for reading in ontology files and annotating natural language with terms from those ontologies. The ontology class is a wrapper for the `pronto` ontology class and contains additional methods for inheriting terms and calculating similarity between sets of terms. The annotation methods include methods for searching for occurences of terms in a larger text and using NOBLE Coder for searching text for terms, and handling and modifying the output from this tool. The main purpose of this package is to allow for reading in ontologies, and using them to produce annotations between instances of text and terms from those ontologies.

**Datasets** (see :doc:`oats.datasets`)

Contains classes and methods for reading in ontology files and annotating natural language with terms from those ontologies. The ontology class is a wrapper for the `pronto` ontology class and contains additional methods for inheriting terms and calculating similarity between sets of terms. The annotation methods include methods for searching for occurences of terms in a larger text and using NOBLE Coder for searching text for terms, and handling and modifying the output from this tool. The main purpose of this package is to allow for reading in ontologies, and using them to produce annotations between instances of text and terms from those ontologies.

**Graphs** (see :doc:`oats.graphs`)

Contains classes and methods for reading in ontology files and annotating natural language with terms from those ontologies. The ontology class is a wrapper for the `pronto` ontology class and contains additional methods for inheriting terms and calculating similarity between sets of terms. The annotation methods include methods for searching for occurences of terms in a larger text and using NOBLE Coder for searching text for terms, and handling and modifying the output from this tool. The main purpose of this package is to allow for reading in ontologies, and using them to produce annotations between instances of text and terms from those ontologies.

**NLP** (see :doc:`oats.nlp`)

Contains classes and methods for reading in ontology files and annotating natural language with terms from those ontologies. The ontology class is a wrapper for the `pronto` ontology class and contains additional methods for inheriting terms and calculating similarity between sets of terms. The annotation methods include methods for searching for occurences of terms in a larger text and using NOBLE Coder for searching text for terms, and handling and modifying the output from this tool. The main purpose of this package is to allow for reading in ontologies, and using them to produce annotations between instances of text and terms from those ontologies.






API Reference Full
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
