from sklearn.metrics import precision_recall_curve
from collections import defaultdict
import itertools
import numpy as np









def classification(graph, id_to_labels):
	"""
	Uses a pairwise similarity graph to obtain some function of the similarities
	between a given node (with unique ID) and all of the labels (such as a group 
	of known similarity or pathway membership, etc).  Currently done by taking
	the average similarity of that node to all the other nodes that were assigned
	that label. Assumes a similarity of 0 to a label group if no other nodes are 
	found to compare a given node to within that label group.
	Args:
	    graph (Graph): Object specifying similarites between all the nodes.
	    id_to_labels (dict): Mapping from node IDs to lists of labels nodes can have.
	Returns:
	    (list,list): List of binary values 0,1 and list of probabilities, same length.
	"""


	# Produce the forward and reverse dictionaries for just the applicable nodes.
	ids_in_graph = graph.get_ids_in_graph()
	id_to_labels = {k:v for (k,v) in id_to_labels.items() if k in ids_in_graph}
	label_to_ids = defaultdict(list)
	for (identifier,labels) in id_to_labels.items():
		for label in labels:
			label_to_ids[label].append(identifier)


	# Generate a predicted value between 0 and 1 for each possible classification (scores).
	# Generate a corresponding list of binary values that indicate the true classification (true).
	labels = label_to_ids.keys()
	ids = id_to_labels.keys()
	y_true = []
	y_scores = []

	for identifier, label in itertools.product(ids,labels):
		y_true.append(int(label in id_to_labels[identifier])) 									# What is the expected response, positive (1) or negative (0)?
		other_ids_with_this_label = [x for x in label_to_ids[label] if x is not identifier] 	# What's the probability assigned to this classification based on the graph?
		if len(other_ids_with_this_label)>0:
			within_label_similarities = np.asarray([graph.get_value(identifier,i) for i in other_ids_with_this_label])
			y_scores.append(np.mean(within_label_similarities))
		else:
			y_scores.append(0.000)

	return(y_true, y_scores)






def pr(y_true, y_scores):
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	return(precision, recall, thresholds)








def consistency_index(graph, id_to_labels):
	"""
	Finds a mapping between labels and the consistency index for that label group,
	which is a measure of how well the graph groups nodes that have that label into
	close proximity within the structure of the graph. Assumes consistency index of 
	0 if there are no within or between label group similarities to evaluate for a 
	given label group.
	Args:
	    graph (Graph): Object specifying similarites between all the nodes.
	    id_to_labels (dict): Mapping from node IDs to lists of labels nodes can have.
	Returns:
	    dict: Mapping between labels and consistency index value as a float.

	"""


	# Produce the forward and reverse dictionaries for just the applicable nodes.
	ids_in_graph = graph.get_ids_in_graph()
	id_to_labels = {k:v for (k,v) in id_to_labels.items() if k in ids_in_graph}
	label_to_ids = defaultdict(list)
	for (identifier,labels) in id_to_labels.items():
		for label in labels:
			label_to_ids[label].append(identifier)


	# TODO figure out if this is a statistically meaningful way to look at the groups.
	# Should probably be accounting for sample size, and whats the distributions look like.

	ids = id_to_labels.keys()
	label_to_consistency_index = {}
	for label in label_to_ids.keys():

		within_label_similarities = []
		between_label_similarities = []
		for identifier in label_to_ids[label]:
			other_ids_with_this_label = [x for x in label_to_ids[label] if x is not identifier]
			other_ids_without_this_label = [x for x in ids if x is not identifier and x not in other_ids_with_this_label]
			within_label_similarities.extend([graph.get_value(identifier,i) for i in other_ids_with_this_label])
			between_label_similarities.extend([graph.get_value(identifier,i) for i in other_ids_without_this_label])

		if len(within_label_similarities)>0 and len(between_label_similarities)>0:
			consistency_index = np.mean(np.asarray(within_label_similarities)) - np.mean(np.asarray(between_label_similarities))
			label_to_consistency_index[label] = consistency_index
		else:
			label_to_consistency_index[label] = 0.000

	return(label_to_consistency_index)






















