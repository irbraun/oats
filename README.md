# OATS (Ontology Annotation and Text Similarity)

### Description

`oats` is a Python package created to provide functions for evaluting similarity between text using a wide variety of approaches including natural language processing, machine learning, and semantic annotation using ontologies. This package was specifically created to be used in a project to assess similarity between phenotype descriptions across plant species in order to better organize this data and make biological predictions. For this reason, some of the subpackages are specifically oriented towards working with biological datasets, although many of the functions are applicable outside of this domain. In general, the purpose of this package is to provide useful wrappers or shortcut functions on top of a number of existing packages and tools for working with biological datasets, ontologies, or natural language in order to make analysis of this kind of data easier. 


### Installing

The package can be installed using the provided wheel with `pip install dist/oats-0.0.1-py3-none-any.whl`.



### Documentation

Full documentation is available on [Read the Docs](https://irbraun-oats.readthedocs.io/en/latest/).


### 1. Biological Datasets

Loading a dataset with plant genes, phenotype descriptions, and ontology terms.
```
>>> from oats.biology.dataset import Dataset
>>> dataset = Dataset("example.csv")
>>> dataset.to_pandas()
```
```
  id  species    unique_gene_identifiers            other_gene_identifiers                                  gene_models                   descriptions                                                                                                                                                                                   annotations                                  sources
----  ---------  ---------------------------------  ------------------------------------------------------  ----------------------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  -------------------------------------------  -----------
   0  ath        ENO1|enolase 1|At1g74030           F2P9_10|AT1G74030.1                                     At1g74030                     Decreased root hair density. Distorted trichomes. Trichomes are less turgescent and are distorted with respect to the wild type. Plants also have fewer root hairs with respect to wild type.  GO:0009507|PO:0001170|GO:0009735             sample data
   1  ath        PIP5K4|At3g56960                   phosphatidyl inositol monophosphate 5 kinase 4|T8M16.6  At3g56960                     Decreased stomatal opening. Delayed stomatal opening.                                                                                                                                          PO:0009046|GO:0009860                        sample data
   2  ath        ALDH10A8|At1g74920                 aldehyde dehydrogenase 10A8|F25A4_11                    At1g74920                     Sensitive to drought. Sensitive to mannitol. Sensitive to salt.                                                                                                                                GO:0005618|GO:0009516|PO:0007123             sample data
   3  zma        nec4|GRMZM5G870342                 cpx1|dks8                                               GRMZM5G870342                 Necrotic leaf. Affected tissue dies. Pale green seedling. Yellow green leaf.                                                                                                                                                                sample data
   4  zma        ufgt2|GRMZM2G117878                UDP-glycosyltransferase 76C1                            GRMZM2G117878                 Salt stress intolerant. Drought susceptible.                                                                                                                                                                                                sample data
   5  zma        ccd8|GRMZM2G446858|Zm00001d043442  Zmccd8|ccd8-trDs|ccd8a                                  GRMZM2G446858|Zm00001d043442  A plant with a thin culm, giving the plant an overall slender appearance. Small ears. Short plant. Slender plant.                                                                              GO:0010311|GO:0022622|GO:1901601|GO:0010016  sample data
```

The `oats.biology.Dataset` contains several methods for adding to, filtering, or accessing the dataset in different ways. This includes getting dictionaries that map the unique internal identifiers to text descriptions or annotations. See the documention for other methods.

```
>>> descriptions = data.get_description_dictionary()
>>> descriptions
```
```
{0: 'Decreased root hair density. Distorted trichomes. Trichomes are less turgescent and are distorted with respect to the wild type. Plants also have fewer root hairs with respect to wild type.',
 1: 'Decreased stomatal opening. Delayed stomatal opening.',
 2: 'Sensitive to drought. Sensitive to mannitol. Sensitive to salt.',
 3: 'Necrotic leaf. Affected tissue dies. Pale green seedling.  Yellow green leaf.',
 4: 'Salt stress intolerant. Drought susceptible.',
 5: 'A plant with a thin culm, giving the plant an overall slender appearance. Small ears. Short plant. Slender plant.'}
```


```
>>> annotations = data.get_description_dictionary()
>>> annotations
```
```
{0: ['GO:0009507', 'PO:0001170', 'GO:0009735'],
 1: ['PO:0009046', 'GO:0009860'],
 2: ['GO:0005618', 'GO:0009516', 'PO:0007123'],
 3: [''],
 4: [''],
 5: ['GO:0010311', 'GO:0022622', 'GO:1901601', 'GO:0010016']}
```







### 2. Ontologies and Annotations


Create an ontology object from reading in an obo file. This class inherits from the `pronto.Ontology` class with some added methods. 
```
>>> from oats.annotation.ontology import Ontology
>>> ont = Ontology("pato.obo")  
```



Getting the label of this particular term.
```
>>> term_id = "PATO:0000587"
>>> ont["PATO:0000587"].name 
```
```
'decreased size'
```



Getting the weight of this term (from 0 to 1) through relative information content.
```
>>> ont.ic(term_id, as_weight=True)
```
```
0.21138654602703566
```


Getting the labels of the terms that are inherited by this term.
```
>>> [ont[t].name for t in ont.inherited(term_id)]
```
```
['size',
 'quality',
 'physical object quality',
 'qualitative',
 'morphology',
 'decreased object quality',
 'deviation (from_normal)',
 'decreased quality',
 'decreased size']
```


Getting the names of the terms that are descendants of this one and their depths in the graph.
```
>>> [(ont[t].name,ont.depth(t)) for t in ont.descendants(term_id)]
```
```
[('decreased thickness', 4),
 ('hypoplastic', 3),
 ('decreased width and length', 5),
 ('decreased diameter', 5),
 ('dwarf-like', 4),
 ('hypotrophic', 5),
 ('decreased area', 4),
 ('decreased perimeter', 5),
 ('dystrophic', 4),
 ('decreased circumference', 6),
 ('decreased length', 4),
 ('decreased volume', 4),
 ('atrophied', 4),
 ('decreased anterior-posterior diameter', 6),
 ('decreased width', 4),
 ('decreased depth', 4),
 ('decreased height', 4),
 ('decreased size', 3)]
```






Annotate descriptions from the dataset with ontology terms using NOBLE Coder.
```
>>> from oats.annotation.annotation import annotate_using_noble_coder
>>> annots = annotate_using_noble_coder(descriptions, "lib/NobleCoder-1.0.jar", "pato", precise=1)
>>> annots
```
```
{0: ['PATO:0001997', 'PATO:0001019', 'PATO:0001617'],
 1: ['PATO:0001997', 'PATO:0000502'],
 2: ['PATO:0000516'],
 3: ['PATO:0001272', 'PATO:0000647', 'PATO:0001941'],
 4: ['PATO:0001152'],
 5: ['PATO:0000592', 'PATO:0000574', 'PATO:0002212', 'PATO:0000569', 'PATO:0000587']}
```



Look at the labels of the terms that were annotated to the descriptions.
```
>>> annot_labels = {i:[ont[t].name for t in terms] for i,terms in annots.items()}
>>> annot_labels
```
```
{0: ['decreased amount', 'mass density', 'deformed'],
 1: ['decreased amount', 'delayed'],
 2: ['sensitive toward'],
 3: ['desaturated green', 'necrotic', 'yellow green'],
 4: ['susceptible toward'],
 5: ['decreased thickness', 'decreased length', 'slender', 'decreased height', 'decreased size']}
```










### 3. Text Distances


Pairwise distance matrix between descriptions using the ontology term annotations assigned above and Jaccard distance.
```
>>> from oats.distances import pairwise
>>> dists = pairwise.with_annotations(annots, ont, metric="jaccard")    
>>> dists.array
```
```
[['0.0000', '0.5556', '0.8750', '0.8182', '0.8824', '0.7600'],
 ['0.5556', '0.0000', '0.9412', '0.9600', '0.9444', '0.8519'],
 ['0.8750', '0.9412', '0.0000', '0.8824', '0.5000', '0.9091'],
 ['0.8182', '0.9600', '0.8824', '0.0000', '0.8889', '0.8966'],
 ['0.8824', '0.9444', '0.5000', '0.8889', '0.0000', '0.9130'],
 ['0.7600', '0.8519', '0.9091', '0.8966', '0.9130', '0.0000']]
```



Pairwise distance matrix using a learned topic model vectors with LDA and using euclidean distance. 
```
>>> dists = pairwise.with_topic_model(descriptions, num_topics=3, algorithm="lda", metric="euclidean")  
>>> dists.array
```
```
[['0.0000', '0.1291', '0.1209', '1.0854', '0.0968', '0.0520'],
 ['0.1291', '0.0000', '0.0082', '0.9738', '0.0323', '0.0771'],
 ['0.1209', '0.0082', '0.0000', '0.9807', '0.0241', '0.0689'],
 ['1.0854', '0.9738', '0.9807', '0.0000', '1.0013', '1.0401'],
 ['0.0968', '0.0323', '0.0241', '1.0013', '0.0000', '0.0448'],
 ['0.0520', '0.0771', '0.0689', '1.0401', '0.0448', '0.0000']]
```


Pairwise distance matrix between descriptions using documenting embeddings inferred from a Doc2Vec model.
```
>>> import gensim
>>> model = gensim.models.Doc2Vec.load("models/doc2vec.bin" )
>>> dists = pairwise.with_doc2vec(descriptions, model=model, metric="cosine")  
>>> dists.array
```
```
[['0.0000', '0.3748', '0.4558', '0.3408', '0.4184', '0.3530'],
 ['0.3748', '0.0000', '0.4455', '0.3657', '0.4021', '0.4193'],
 ['0.4558', '0.4455', '0.0000', '0.4565', '0.4158', '0.4606'],
 ['0.3408', '0.3657', '0.4565', '0.0000', '0.4192', '0.3084'],
 ['0.4184', '0.4021', '0.4158', '0.4192', '0.0000', '0.4539'],
 ['0.3530', '0.4193', '0.4606', '0.3084', '0.4539', '0.0000']]
```


Getting the first few values of the vector represention of a particular description in the dataset.
```
>>> vector = dists.vector_dictionary[3]
>>> vector[:4]
```
```
['0.0548', '-0.1059', '-0.0027', '0.1615']
```






### References
This package makes use of several underlying methods, libraries, and resources.

- NLP Libraries and Models ([Gensim](https://radimrehurek.com/gensim/), [NLTK](https://www.nltk.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Word2Vec](https://arxiv.org/abs/1301.3781), [Doc2Vec](https://arxiv.org/abs/1405.4053))
- Ontologies and Annotation ([NOBLE Coder](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0871-y), [Pronto](https://github.com/althonos/pronto))
- Biological Databases ([STRING](https://string-db.org/), [KEGG](https://www.genome.jp/kegg/), [PlantCyc](https://www.plantcyc.org/))


 




### Feedback
This package is a work in progress. Send any feedback, questions, or suggestions to irbraun at iastate dot edu.
