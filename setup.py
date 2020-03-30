import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='oats',  
    version='0.0.1',
    author="Ian Braun",
    author_email="irbraun@iastate.edu",
    description="Ontology Annotation and Text Similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irbraun/oats",
    packages=setuptools.find_packages(),
    install_requires=[
        "pywsd == 1.2.4",
        "gensim == 3.8.1",
        "fastsemsim == 1.0.0",
        "pandas == 1.0.0",
        "numpy == 1.16.4",
        "scikit-learn == 0.21.2",
        "rdflib == 4.2.2",
        "sparqlwrapper == 1.8.2",
        "rapidfuzz == 0.3.0",
        "nltk == 3.4.4",
        "pronto == 0.12.2",
        "biopython == 1.76"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
