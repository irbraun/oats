# OATS (Ontology Annotation and Text Similarity)

### Description

`oats` is a python package created to provide functions for evaluting similarity between text using a wide variety of approaches including natural language processing, machine learning, and semantic annotation using ontologies. This package was specifically created to be used in a project to assess similarity between phenotype descriptions across plant species in order to better organize this data and make biological predictions. For this reason, some of the subpackages are specifically oriented towards working with biological datasets, although many of the functions are applicable outside of this domain. In general, the purpose of this package is to provide useful wrappers or shortcut functions on top of a number of existing packages and tools for working with biological datasets, ontologies, or natural language in order to make analysis of this kind of data easier. 


### Setup

Use pip to install with
```
$ pip install dist/oats-0.0.1-py3-none-any.whl
```



### Documentation

Available on Read the Docs [here](https://irbraun-oats.readthedocs.io/en/latest/).





### 1. Biological Datasets

Loading a dataset with plant genes, phenotype descriptions, and ontology terms.
```
>>> from oats.biology.dataset import Dataset
>>> data = Dataset("example.csv")
>>> data.to_pandas()
```
```
  id  species    gene_names           gene_synonyms                                   description                                                                            term_ids               sources
----  ---------  -------------------  ----------------------------------------------  -------------------------------------------------------------------------------------  ---------------------  ---------
   0  ath        At1g74030            F2P9_10                                         Decreased root hair density. Distorted trichomes.                                      GO:0009507|PO:0001170  example
   1  ath        At1g74030|ENO1       F2P9_10|AT1G74030.1                             Trichomes are less turgescent and are distorted with respect to the wild type.         GO:0009735             example
   2  ath        ENO1|enolase 1       F2P9_10|AT1G74030.1                             Plants also have fewer root hairs with respect to wild type.                           GO:0009735             example
   3  ath        At3g56960|PIP5K4     phosphatidyl inositol monophosphate 5 kinase 4  Decreased stomatal opening.                                                            PO:0009046             example
   4  ath        At3g56960            T8M16.6                                         Delayed stomatal opening.                                                              GO:0009860             example
   5  ath        At1g74920|ALDH10A8   aldehyde dehydrogenase 10A8                     Sensitive to drought. Sensitive to mannitol.                                           GO:0005618|GO:0009516  example
   6  ath        At1g74920            F25A4_11                                        Sensitive to salt.                                                                     PO:0007123             example
   7  zma        nec4|GRMZM5G870342   nec4|cpx1                                       Necrotic leaf. Affected tissue dies.                                                                          example
   8  zma        GRMZM5G870342        cpx1|dks8                                       Pale green seedling.  Yellow green leaf.                                                                      example
   9  zma        GRMZM2G117878|ufgt2                                                  Salt stress intolerant.                                                                                       example
  10  zma        GRMZM2G117878        UDP-glycosyltransferase 76C1                    Drought susceptible.                                                                                          example
  11  zma        ccd8                 Zmccd8                                          A plant with a thin culm, giving the plant an overall slender appearance. Small ears.  GO:0010311|GO:0022622  example
  12  zma        ccd8|GRMZM2G446858   ccd8-trDs|ccd8                                  Short plant.                                                                           GO:1901601             example
  13  zma        ccd8|Zm00001d043442  ccd8|ccd8a                                      Slender plant.                                                                         GO:0010016             example
```





Merging the dataset so that each row and unique ID represents a single gene.
```
>>> data.collapse_by_all_gene_names()
>>> data.to_pandas()
```
```
  id  species    gene_names                         gene_synonyms                                           description                                                                                                                                                                                    term_ids                                     sources
----  ---------  ---------------------------------  ------------------------------------------------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  -------------------------------------------  ---------
   0  ath        At1g74030|ENO1|enolase 1           F2P9_10|AT1G74030.1                                     Decreased root hair density. Distorted trichomes. Trichomes are less turgescent and are distorted with respect to the wild type. Plants also have fewer root hairs with respect to wild type.  GO:0009507|PO:0001170|GO:0009735             example
   1  ath        At3g56960|PIP5K4                   phosphatidyl inositol monophosphate 5 kinase 4|T8M16.6  Decreased stomatal opening. Delayed stomatal opening.                                                                                                                                          PO:0009046|GO:0009860                        example
   2  ath        At1g74920|ALDH10A8                 aldehyde dehydrogenase 10A8|F25A4_11                    Sensitive to drought. Sensitive to mannitol. Sensitive to salt.                                                                                                                                GO:0005618|GO:0009516|PO:0007123             example
   3  zma        nec4|GRMZM5G870342                 nec4|cpx1|dks8                                          Necrotic leaf. Affected tissue dies. Pale green seedling.  Yellow green leaf.                                                                                                                                                               example
   4  zma        GRMZM2G117878|ufgt2                UDP-glycosyltransferase 76C1                            Salt stress intolerant. Drought susceptible.                                                                                                                                                                                                example
   5  zma        ccd8|GRMZM2G446858|Zm00001d043442  Zmccd8|ccd8-trDs|ccd8|ccd8a                             A plant with a thin culm, giving the plant an overall slender appearance. Small ears. Short plant. Slender plant.                                                                              GO:0010311|GO:0022622|GO:1901601|GO:0010016  example
```




Relating those unique IDs for each gene to biochemical pathways from PlantCyc.
```
>>> from oats.biology.groupings import Groupings
>>> pathway_species_files = {"ath":"aracyc_pathways.20180702", "zma":"corncyc_pathways.20180702"}
>>> groupings = Groupings(pathway_species_files, "plantcyc")
>>> id_to_groups, group_to_ids = groupings.get_groupings_for_dataset(data)
>>> id_to_groups
```
```
{0: ['PWY-1042', 'PWY-5723', 'PWY66-399', 'PWY-5484', 'GLUCONEO-PWY', 'GLYCOLYSIS'],
 1: ['PWY-6351', 'PWY-6352'],
 2: ['PWY-2', 'PWY1F-353'],
 3: [],
 4: [],
 5: ['PWY-7101', 'PWY-6806']}
```





### 2. Ontologies and Annotations


Get a dictionary that relates unique IDs to phenotype descriptions.
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



Create an ontology object and annotate descriptions with terms using NOBLE Coder.
```
>>> from oats.annotation.ontology import Ontology
>>> from oats.annotation.annotation import annotate_using_noble_coder
>>>
>>> ont = Ontology("pato.obo")    
>>> annots = annotate_using_noble_coder(descriptions, "NobleCoder-1.0.jar", "pato", precise=1)
>>> annots
```
```
{0: ['PATO:0001997', 'PATO:0001617', 'PATO:0001019'],
 1: ['PATO:0001997', 'PATO:0000502'],
 2: ['PATO:0000516'],
 3: ['PATO:0001941', 'PATO:0000647', 'PATO:0001272'],
 4: ['PATO:0001152'],
 5: ['PATO:0002212', 'PATO:0000587', 'PATO:0000569', 'PATO:0000574', 'PATO:0000592']}
```



Look at the labels of the terms that were annotated to the descriptions.
```
>>> annot_labels = {i:[ont.get_label_from_id(t) for t in terms] for i,terms in annots.items()}
>>> annot_labels
```
```
{0: ['decreased amount', 'deformed', 'mass density'],
 1: ['decreased amount', 'delayed'],
 2: ['sensitive toward'],
 3: ['yellow green', 'necrotic', 'desaturated green'],
 4: ['susceptible toward'],
 5: ['slender', 'decreased size', 'decreased height', 'decreased length', 'decreased thickness']}
```


### 3. Text Distances


Pairwise distance matrix between descriptions using the ontology term annotations assigned above.
```
>>> from oats.distances import pairwise as pw
>>> dists = dists = pw.pairwise_square_annotations(annots, ont, metric="jaccard")  
>>> dists.array
```
```
[['0.0000', '0.5714', '0.8333', '0.7500', '0.8462', '0.6842'],
 ['0.5714', '0.0000', '0.9286', '0.9000', '0.9333', '0.8182'],
 ['0.8333', '0.9286', '0.0000', '0.8462', '0.3333', '0.8889'],
 ['0.7500', '0.9000', '0.8462', '0.0000', '0.8571', '0.8696'],
 ['0.8462', '0.9333', '0.3333', '0.8571', '0.0000', '0.8947'],
 ['0.6842', '0.8182', '0.8889', '0.8696', '0.8947', '0.0000']]
```



Pairwise distance matrix between descriptions using n-grams.
```
>>> dists = pw.pairwise_square_ngrams(ids_to_texts=descriptions, binary=True, metric="jaccard")    
>>> dists.array
```
```
[['0.0000', '0.9583', '0.9600', '1.0000', '1.0000', '0.9375'],
 ['0.9583', '0.0000', '1.0000', '1.0000', '1.0000', '1.0000'],
 ['0.9600', '1.0000', '0.0000', '1.0000', '0.7500', '1.0000'],
 ['1.0000', '1.0000', '1.0000', '0.0000', '1.0000', '1.0000'],
 ['1.0000', '1.0000', '0.7500', '1.0000', '0.0000', '1.0000'],
 ['0.9375', '1.0000', '1.0000', '1.0000', '1.0000', '0.0000']]
```


Pairwise distance matrix between descriptions using topic modeling.
```
>>> dists = pw.pairwise_square_topic_model(ids_to_texts=descriptions, num_topics=3, algorithm="lda", metric="euclidean")  
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


Pairwise distance matrix between descriptions using a Doc2Vec model.
```
>>> import gensim
>>> model = gensim.models.Doc2Vec.load("doc2vec.bin" )
>>> dists = pw.pairwise_square_doc2vec(model=model, ids_to_texts=descriptions, metric="cosine")    
>>> dists.array
```
```
[['0.0000', '0.3742', '0.4558', '0.3409', '0.4186', '0.3525'],
 ['0.3742', '0.0000', '0.4461', '0.3658', '0.4026', '0.4185'],
 ['0.4558', '0.4461', '0.0000', '0.4557', '0.4146', '0.4586'],
 ['0.3409', '0.3658', '0.4557', '0.0000', '0.4205', '0.3088'],
 ['0.4186', '0.4026', '0.4146', '0.4205', '0.0000', '0.4536'],
 ['0.3525', '0.4185', '0.4586', '0.3088', '0.4536', '0.0000']]
```

### References
This package makes use of several underlying methods, libraries, and resources.

- NLP Libraries and Models ([Gensim](https://radimrehurek.com/gensim/), [NLTK](https://www.nltk.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Word2Vec](https://arxiv.org/abs/1301.3781), [Doc2Vec](https://arxiv.org/abs/1405.4053))
- Ontologies and Annotation ([NOBLE Coder](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0871-y), [Pronto](https://github.com/althonos/pronto))
- Biological Databases ([STRING](https://string-db.org/), [KEGG](https://www.genome.jp/kegg/), [PlantCyc](https://www.plantcyc.org/))


 




### Feedback
This package is a work in progress. Send any feedback, questions, or suggestions to irbraun at iastate dot edu.
