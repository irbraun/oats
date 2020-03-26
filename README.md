# OATS (Ontology Annotation and Text Similarity)

### Description

`oats` is a python package created to provide functions for quickly generating similarity values between text using a wide variety of approaches from natural language processing, machine learning, and semantic annotation with ontologies. Specifically, this package is aimed at generating pairwise similarity matrices for large datasets of short descriptions, although there is flexability. The package was created to support an analysis of similarities between phenotype descriptions and 



This is a python package created to provide functions for generating and working with networks constructed from natural language descriptions. Specifically, this package was designed for datasets of text descriptions associated with genes, for the purposes of predicting gene function or membership in pathways and regulatory networks through constructing similarity networks with genes and their associated descriptions as nodes. However, many of the functions provided here can generalize to any problem requiring constructing similarity networks based on natural language datasets.

### Documentation

Available on Read the Docs [here][1].

#### BioData
The `oats.biodata` subpackage has classes, methods, and functions for handling, sorting, merging, and other things data that has to do with genes mapped to phenotype descriptions mapped to ontology term annotations and other things like that.


```
>>> from oats.datasets.dataset import Dataset
>>> data = Dataset("example.csv")
>>> data.to_pandas()
```






```
  id  species    gene_names           gene_synonyms                                   description                                                                            term_ids               sources
----  ---------  -------------------  ----------------------------------------------  -------------------------------------------------------------------------------------  ---------------------  ---------
   0  ath        At1g74030            F2P9_10                                         Decreased root hair density. Distorted trichomes.                                      GO:0009507|PO:0001170
   1  ath        At1g74030|ENO1       F2P9_10|AT1G74030.1                             Trichomes are less turgescent and are distorted with respect to the wild type.         GO:0009735
   2  ath        ENO1|enolase 1       F2P9_10|AT1G74030.1                             Plants also have fewer root hairs with respect to wild type.                           GO:0009735
   3  ath        At3g56960|PIP5K4     phosphatidyl inositol monophosphate 5 kinase 4  Decreased stomatal opening.                                                            PO:0009046
   4  ath        At3g56960            T8M16.6                                         Delayed stomatal opening.                                                              GO:0009860
   5  ath        At1g74920|ALDH10A8   aldehyde dehydrogenase 10A8                     Sensitive to drought. Sensitive to mannitol.                                           GO:0005618|GO:0009516
   6  ath        At1g74920            F25A4_11                                        Sensitive to salt.                                                                     PO:0007123
   7  zma        nec4|GRMZM5G870342   nec4|cpx1                                       Necrotic leaf. Affected tissue dies.
   8  zma        GRMZM5G870342        cpx1|dks8                                       Pale green seedling.  Yellow green leaf.
   9  zma        GRMZM2G117878|ufgt2                                                  Salt stress intolerant.
  10  zma        GRMZM2G117878        UDP-glycosyltransferase 76C1                    Drought susceptible.
  11  zma        ccd8                 Zmccd8                                          A plant with a thin culm, giving the plant an overall slender appearance. Small ears.  GO:0010311|GO:0022622
  12  zma        ccd8|GRMZM2G446858   ccd8-trDs|ccd8                                  Short plant.                                                                           GO:1901601
  13  zma        ccd8|Zm00001d043442  ccd8|ccd8a                                      Slender plant.                                                                         GO:0010016
```



```
>>> data.collapse_by_all_gene_names()
>>> data.to_pandas()
```



```
  id  species    gene_names                         gene_synonyms                                           description                                                                                                                                                                                    term_ids                                     sources
----  ---------  ---------------------------------  ------------------------------------------------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  -------------------------------------------  ---------
   0  ath        At1g74030|ENO1|enolase 1           F2P9_10|AT1G74030.1                                     Decreased root hair density. Distorted trichomes. Trichomes are less turgescent and are distorted with respect to the wild type. Plants also have fewer root hairs with respect to wild type.  GO:0009507|PO:0001170|GO:0009735
   1  ath        At3g56960|PIP5K4                   phosphatidyl inositol monophosphate 5 kinase 4|T8M16.6  Decreased stomatal opening. Delayed stomatal opening.                                                                                                                                          PO:0009046|GO:0009860
   2  ath        At1g74920|ALDH10A8                 aldehyde dehydrogenase 10A8|F25A4_11                    Sensitive to drought. Sensitive to mannitol. Sensitive to salt.                                                                                                                                GO:0005618|GO:0009516|PO:0007123
   3  zma        nec4|GRMZM5G870342                 nec4|cpx1|dks8                                          Necrotic leaf. Affected tissue dies. Pale green seedling.  Yellow green leaf.
   4  zma        GRMZM2G117878|ufgt2                UDP-glycosyltransferase 76C1                            Salt stress intolerant. Drought susceptible.
   5  zma        ccd8|GRMZM2G446858|Zm00001d043442  Zmccd8|ccd8-trDs|ccd8|ccd8a                             A plant with a thin culm, giving the plant an overall slender appearance. Small ears. Short plant. Slender plant.                                                                              GO:0010311|GO:0022622|GO:1901601|GO:0010016
```






```
>>> from oats.datasets.groupings import Groupings
>>> pathway_species_files = {
>>>   "ath":"../data/group_related_files/pmn/aracyc_pathways.20180702", 
>>>   "zma":"../data/group_related_files/pmn/corncyc_pathways.20180702"}
>>> groupings = Groupings(pathway_species_files, "pmn")
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

#### Annotations
The `oats.annotations` subpackage has classes, methods, and functions for handling, sorting, merging, and other things data that has to do with genes mapped to phenotype descriptions mapped to ontology term annotations and other things like that. In addition to using the NOBLE Coder annotation jarfile, there are other methods available such as direct or fuzzy text alignment. The ontology class also has methods and some static functions for handling ontology related tasks.


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
>>> from oats.annotation.ontology import Ontology
>>> from oats.annotation.annotation import annotate_using_noble_coder
>>>
>>> ont = Ontology("pato.obo") 
>>> noblecoder_jarfile_path = "../lib/NobleCoder-1.0.jar"      
>>> annots = annotate_using_noble_coder(descriptions, "../lib/NobleCoder-1.0.jar", "pato", precise=1)
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

```
>>> annot_labels = {i:[ont.get_label_from_id(t) for t in term_list] for i,term_list in annots.items()}
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


#### Distances
The `oats.distances` subpackage has classes, methods, and functions for handling, sorting, merging, and other things data that has to do with genes mapped to phenotype descriptions mapped to ontology term annotations and other things like that. In addition to using the NOBLE Coder annotation jarfile, there are other methods available such as direct or fuzzy text alignment. The ontology class also has methods and some static functions for handling ontology related tasks. Talk about what the options are here for finding distanes matrices between stuff. 


```
>>> from oats.distances.pairwise import pairwise_square_word2vec
>>> from oats.distances.pairwise import pairwise_square_doc2vec
>>> from oats.distances.pairwise import pairwise_square_bert
>>> from oats.distances.pairwise import pairwise_square_ngrams
>>> from oats.distances.pairwise import pairwise_square_topics
>>>
>>> doc2vec_model = gensim.doc2vec.load(model="models/doc2vec.bin")
>>> word2vec_model = gensim.word2vec.load(model="models/word2vec_enwiki.bin")
>>> descriptions = data.get_descriptions_dictionary()
>>> pprint(descriptions)

0: a description of something
1: a descriptions of something else
2: a description of something else

>>> dists = pairwise_square_word2vec(model=m, id_to_texts=descriptions)
>>> dists.matrix

[[0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453]]

>>> dists = pairwise_square_ngrams(model=m, id_to_texts=descriptions)
>>> dists.matrix

[[0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453]]

>>> dists = pairwise_square_topics(model=m, id_to_texts=descriptions)
>>> dists.matrix

[[0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453],
 [0.245, 0.235, 0.345, 0.3453]]

```





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





[1]: https://irbraun-oats.readthedocs.io/en/latest/
