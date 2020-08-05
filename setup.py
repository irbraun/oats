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
        "pandas == 1.0.*",
        "numpy == 1.18.*",
        "nltk == 3.4.*",
        "pywsd == 1.2.*",
        "scikit-learn == 0.22.*",
        "biopython == 1.76.*",
        "more-itertools == 8.4.*",
        "networkx == 2.4.*",
        "scipy == 1.5.*",
        "pytorch-pretrained-bert==0.6.2",
        "pronto == 2.1.0",
        "gensim == 3.8.1",
        "rapidfuzz == 0.3.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
