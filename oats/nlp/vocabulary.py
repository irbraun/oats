from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from pywsd.lesk import cosine_lesk
from scipy.stats import fisher_exact
import itertools
import itertools
import numpy as np
import pandas as pd
import networkx as nx


from oats.utils.utils import flatten
from oats.nlp.preprocess import get_clean_token_list






def get_wordnet_related_words_from_word(word, context, synonyms=1, hypernyms=0, hyponyms=0):
	"""
	Method to generate a list of words that are found to be related to the input word through
	the WordNet ontology and resource. The correct sense of the input word to be used within the
	context of WordNet is picked based on disambiguation from the PyWSD package which takes
	the surrounding text (or whatever text is provided as context) into account. All synonyms,
	hypernyms, and hyponyms are considered to be related words in this case.
	
	Args:
		word (str): The word for which we want to find related words.

		context (str): Text to use for word-sense disambigutation, usually sentence the word is in.

		synonyms (int, optional): Set to 1 to include synonyms in the set of related words.
		
		hypernyms (int, optional): Set to 1 to included hypernyms in the set of related words.
		
		hyponyms (int, optional): Set to 1 to include hyponyms in the set of related words.
	
	Returns:
		list: The list of related words that were found, could be empty if nothing was found.
	"""

	# To get the list of synsets for this word if not using disambiguation.
	list_of_possible_s = wordnet.synsets(word)

	# Disambiguation of synsets (https://github.com/alvations/pywsd).
	# Requires installation of non-conda package PyWSD from pip ("pip install pywsd").
	# The methods of disambiguation that are supported by this package are: 
	# (simple_lesk, original_lesk, adapted_lesk, cosine_lesk, and others). 
	s = cosine_lesk(context, word)

	try:
		# Generate related words using wordnet, including synonyms, hypernyms, and hyponyms.
		# The lists of hypernyms and hyponyms need to be flattened because they're lists of lists from synsets.
		# definition() yields a string. 
		# lemma_names() yields a list of strings.
		# hypernyms() yields a list of synsets.
		# hyponyms() yields a list of synsets.
		synset_definition = s.definition()
		synonym_lemmas = s.lemma_names() 													
		hypernym_lemmas_nested_list = [x.lemma_names() for x in s.hypernyms()] 
		hyponym_lemmas_nested_list = [x.lemma_names() for x in s.hyponyms()]
		# Flatten those lists of lists.
		hypernym_lemmas = list(itertools.chain.from_iterable(hypernym_lemmas_nested_list))
		hyponym_lemmas = list(itertools.chain.from_iterable(hyponym_lemmas_nested_list))

		# Print out information about the synset that was picked during disambiguation.
		#print(synset_definition)
		#print(synonym_lemmas)
		#print(hypernym_lemmas)
		#print(hyponym_lemmas)
		related_words = []
		if synonyms==1:
			related_words.extend(synonym_lemmas)
		if hypernyms==1:
			related_words.extend(hypernym_lemmas)
		if hyponyms==1:
			related_words.extend(hyponym_lemmas)
		return(related_words)

	except AttributeError:
		return([])





def get_word2vec_related_words_from_word(word, model, threshold, max_qty):
	"""
	Method to generate a list of words that are found to be related to the input word through
	assessing similarity to other words in a word2vec model of word embeddings. The model can
	be learned from relevant text data or can be pre-trained on an existing source. All words
	that satisfy the threshold provided up to the quantity specified as the maximum are added.
	
	Args:
		word (str): The word for which we want to find other related words.
		
		model (Word2Vec): The actual model object that has already been loaded.
		
		threshold (float): Similarity threshold that must be satisfied to add a word as related.
		
		max_qty (int): Maximum number of related words to accept.
	
	Returns:
		list: The list of related words that were found, could be empty if nothing was found.
	"""
	
	related_words = []
	try:
		matches = model.most_similar(word, topn=max_qty)
	except KeyError:
		matches = []
	for match in matches:
		word_in_model = match[0]
		similarity = match[1]
		if (similarity >= threshold):
			related_words.append(word_in_model)
	return(related_words)








def get_wordnet_related_words_from_doc(description, synonyms=1, hyperhyms=0, hyponyms=1):
	"""
	Get a dictionary mapping tokens in some text to related words found with WordNet.
	Note that these could not only be synonyms but also hypernyms and hyponyms depending
	on what parameters are used.
	
	Args:
		description (str): Any string of text, a description of something.
		
		synonyms (int, optional): Set to 1 to include synonyms in the set of related words.
		
		hypernyms (int, optional): Set to 1 to included hypernyms in the set of related words.
		
		hyponyms (int, optional): Set to 1 to include hyponyms in the set of related words.
	
	Returns:
		dict: A mapping from a string to a list of strings, the found related words..
	"""
	tokens = get_clean_token_list(description)
	synonym_dict = {token:get_wordnet_related_words_from_word(token, description, synonyms, hypernyms, hyponyms) for token in tokens}
	return(synonym_dict)





def get_word2vec_related_words_from_doc(description, model, threshold, max_qty):
	"""
	Get a dictionary mapping tokens in some text to related words found with Word2Vec.
	Note that these are not necessarily truly synonyms, but may just be words that are 
	strongly or weakly related to a given word, depending on how strict the threshold
	parameters are that are used.
	
	Args:
		description (str): Any string of text, a description of something.
		
		model (Word2Vec): The actual model object that has already been loaded.
		
		threshold (float): Similarity threshold that must be satisfied to add a word as related.
		
		max_qty (int): Maximum number of related words to accept for a single token.
	
	Returns:
		dict: A mapping from a string to a list of strings, the found related words..
	"""
	tokens = get_clean_token_list(description)
	synonym_dict = {token:get_word2vec_related_words_from_word(token, model, threshold, max_qty) for token in tokens}
	return(synonym_dict)







def get_vocab_from_tokens(tokens):
	"""
	Generates a mapping between each token and some indices 0 to n that can place
	that token at a particular index within a vector. This is a vocabulary dict 
	that is in the format necessary for passing as an argument to the sklearn
	classes for generating feature vectors from input text.
	
	Args:
		tokens (list): A list of tokens that belong in the vocabulary.
	
	Returns:
		dict: A mapping between each token and an index from zero to n.
	"""
	vocab_dictionary = {token:i for i,token in enumerate(list(tokens))}
	return(vocab_dictionary)








def get_overrepresented_tokens(interesting_text, background_text, max_features):
	"""
	See https://liferay.de.dariah.eu/tatom/feature_selection.html.
	This way uses the difference in the rate of each particular words between the 
	interesting text and the background text to determine what the vocabulary of 
	relevant words should be. This means we are selecting as features things that are 
	important in some paritcular domain but are not as important in the general
	language. This is potentially one method of finding words which will be of removing
	the general words from the text that is parsed for a particular domain. Potential 
	problem is that we actually want words (features) that are good at differentiating 
	different phenotypes, which is a slightly different question. 
	
	Args:
		interesting_text (str): A string of many tokesn coming form examples of interest.
		
		background_text (str): A string of many tokens coming from some background examples.
		
		max_features (int): The maximum number of features (tokens) in the returned vocabulary.
	
	Returns:
		list: A list of features which are tokens, or words, which are represent
	"""
	vectorizer = CountVectorizer(input='content')
	dtm = vectorizer.fit_transform([interesting_text,background_text])
	vocab = np.array(vectorizer.get_feature_names())
	dtm = dtm.toarray()
	rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)
	keyness = rates[0]-rates[1]
	ranking = np.argsort(keyness)[::-1]
	vocab_tokens = vocab[ranking][0:max_features]
	return(vocab_tokens)








def reduce_vocab_connected_components(descriptions, tokens, distance_matrix, threshold):
	"""
	Reduces the vocabulary size for a dataset of provided tokens by looking at a provided 
	distance matrix between all the words and creating new tokens to represent groups of 
	words that have a small distance (less than the threshold) between two of the members
	of that group. This problem is solved here as a connected components problem by creating
	a graph where tokens are words, and each word is connected to itself and any word where
	the distance to that word is less than the threshold. Note that the Linares Pontes
	algorithm is generally favorable to this approach because if the threshold is too high
	the connected components can quickly become very large.
	
	Args:
		descriptions (dict): A mapping between IDs and text descriptions.

		tokens (list): A list of tokens from which to construct the vocabulary.

		distance_matrix (np.array): An by n square matrix of distances where n must be length of tokens list and indices must correspond.

		threshold (float): The value where a distance of less than this threshold indicates the words should be collapsed to a new token.
	
	Returns:
		dict: Mapping between IDs and text descriptions with reduced vocabulary, matches input.

		dict: Mapping between tokens present in the original texts and corresponding tokens from the reduced vocabulary.

		dict: Mapping between tokens present in the reduced vocab and lists of corresponding original vocabulary tokens.
	"""

	g = nx.Graph()
	edges = []
	node_id_to_token = {str(i):token for i,token in enumerate(tokens)}
	token_to_node_id = {token:str(i) for i,token in node_id_to_token.items()}

	# Add the self edges between each token, then edges that satisfy the distance threshold.
	edges.extend([(token_to_node_id[token],token_to_node_id[token]) for token in tokens]) 
	for i,j in itertools.product(np.arange(distance_matrix.shape[0]),np.arange(distance_matrix.shape[0])):
		if distance_matrix[i,j] < threshold:
			edges.append((token_to_node_id[tokens[i]],token_to_node_id[tokens[j]]))   

	# Add all the edges from the graph and build the dictionaries by looking at connected components.
	g.add_edges_from(edges)
	components = nx.connected_components(g)
	transform_dict = {}
	untransform_dict = defaultdict(list)
	new_token_ctr = 0
	for c in components:
		if len(c) > 1:
			for node_id in c:
				word = node_id_to_token[node_id]
				new_token = "TOKEN_{}".format(new_token_ctr)
				transform_dict[word] = new_token
				untransform_dict[new_token].append(word)
			new_token_ctr = new_token_ctr+1
		else:
			word = node_id_to_token[list(c)[0]]
			transform_dict[word] = word
			untransform_dict[word] = [word]
	
	# Do the replacements in each input description and return the modified dictionary of them.
	reduced_descriptions = {}
	for i,description in descriptions.items():
		reduced_description = " ".join([transform_dict[token] for token in description.split()])
		reduced_descriptions[i] = reduced_description
	return(reduced_descriptions, transform_dict, untransform_dict)













def reduce_vocab_linares_pontes(descriptions, tokens, distance_matrix, n):
	"""
	Implementation of the algorithm described in the paper cited below. In short, this returns the 
	descriptions with each word replaced by the most frequently used token in the set of tokens that 
	consists of that word and the n most similar words as given by the distance matrix provided. Some 
	values of n that are used in the papers are 1, 2, and 3. Note that the descriptions in the passed 
	in dictionary should already be preprocessed in whatever way is necessary, but they should atleast
	be formatted as lowercase tokens that are separated by a single space in each description. The
	tokens in the list of tokens should be pulled directly from those descriptions and be found
	by splitting by a single space. They are passed in as a separate list though because the index
	of the token in the list has to correspond to the index of that token in the distance matrix. 
	If the descriptions contain any tokens that are not present in the tokens list will not be affected
	when altering the tokens that are present in the descriptions.

	Elvys Linhares Pontes, Stéphane Huet, Juan-Manuel Torres-Moreno, Andréa Carneiro Linhares. 
	Automatic Text Summarization with a Reduced Vocabulary Using Continuous Space Vectors. 
	21st International Conference on Applications of Natural Language to Information Systems (NLDB),
	2016, Salford, United Kingdom. pp.440-446, ff10.1007/978-3-319-41754-7_46ff. ffhal-01779440
	
	Args:
		descriptions (dict): A mapping between IDs and text descriptions.
		
		tokens (list): A list of strings which are tokens that appear in the descriptions. 
		
		distance_matrix (np.array): A square array of distances between the ith and jth token in the tokens list. 
		
		n (int): The number of most similar words to consider when replacing a word in building the reduced vocabulary.
	
	Returns:
		dict: Mapping between IDs and text descriptions with reduced vocabulary, matches input.

		dict: Mapping between tokens present in the original vocab and the token it is replaced with in the reduced vocabulary.

		dict: Mapping between tokens present in the reduced vocab and lists of corresponding original vocabulary tokens.
	"""

	# Find the frequency distribution of all of the tokens in the passed in descriptions.
	fdist = FreqDist(flatten([[token for token in description.split()] for description in descriptions.values()]))
	token_to_index = {token:i for i,token in enumerate(tokens)}
	index_to_token = {i:token for i,token in enumerate(tokens)}
	
	# Create a mapping between all the tokens and the words they'll be replaced with.
	token_to_reduced_vocab_token = {}
	reduced_vocab_token_to_tokens = defaultdict(list)
	for token in tokens:
		index = token_to_index[token]
		n_indices = np.argpartition(distance_matrix[index], n)[:n]
		n_tokens = [index_to_token[idx] for idx in list(n_indices)]
		n_frequencies = [fdist[token] for token in n_tokens]
		maxfreq_token = n_tokens[np.argmax(n_frequencies)]
		token_to_reduced_vocab_token[token] = maxfreq_token
		reduced_vocab_token_to_tokens[maxfreq_token].append(token)

		
	# Do the replacements in each input description and return the modified dictionary of them.
	reduced_descriptions = {}
	for i,description in descriptions.items():
		#reduced_description = " ".join([token_to_reduced_vocab_token[token] for token in description.split()])
		reduced_description = " ".join([token_to_reduced_vocab_token.get(token,token) for token in description.split()])
		reduced_descriptions[i] = reduced_description
	return(reduced_descriptions, token_to_reduced_vocab_token, reduced_vocab_token_to_tokens)
			








def token_enrichment(all_ids_to_texts, group_ids):
    """ Obtain a dataframe with the results of a token enrichment analysis using Fisher exact test with the results sorted by p-value.
    
    Args:
        all_ids_to_texts (dict of int:str): A mapping between unique integer IDs (for genes) and some string of text.

        group_ids (list of int): The IDs which should be a subset of the dictionary argument that refer to those belonging to the group to be tested.
    
    Returns:
        pandas.DataFrame: A dataframe sorted by p-value that contains the results of the enrichment analysis with one row per token.
    """
    
    # Tokenize the strings of text to identify individual words and find all the unique tokens that appear anywhere in the texts.
    all_ids_to_token_lists = {i:word_tokenize(text) for i,text in all_ids_to_texts.items()}
    unique_tokens = list(set(flatten(all_ids_to_token_lists.values())))
    
    # For each token, determine the total number of texts that it is present in.
    num_ids_with_token_t = lambda t,id_to_tokens: [(t in tokens) for i,tokens in id_to_tokens.items()].count(True) 
    token_to_gene_count = {t:num_ids_with_token_t(t,all_ids_to_token_lists) for t in unique_tokens}
    total_num_of_genes = len(all_ids_to_token_lists)
    df = pd.DataFrame(unique_tokens, columns=["token"])
    df["genes_with"] = df["token"].map(lambda x: token_to_gene_count[x])
    df["genes_without"] = total_num_of_genes-df["genes_with"] 
    
    # For each token, determine the total number of texts that belong to the group to be tested that it is present in.
    num_of_genes_in_group = len(group_ids)
    ids_in_group_to_token_lists = {i:tokens for i,tokens in all_ids_to_token_lists.items() if i in group_ids}
    token_to_gene_in_group_count = {t:num_ids_with_token_t(t,ids_in_group_to_token_lists) for t in unique_tokens}
    df["group_genes_with"] = df["token"].map(lambda x: token_to_gene_in_group_count[x])
    df["group_genes_without"] = num_of_genes_in_group-df["group_genes_with"] 
    
    # Using those values, perform the Fisher exact test to obtain a p-value for each term, sort the results, and return.
    df["p_value"] = df.apply(lambda row: fisher_exact([[row["group_genes_with"],row["genes_with"]],[row["group_genes_without"],row["genes_without"]]])[1], axis=1)
    df.sort_values(by="p_value", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return(df)
    


	











