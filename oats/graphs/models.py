from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from functools import reduce
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier









def apply_weights(df, predictor_columns, weights_dict):
	"""
	Generates a dataframe with a single value column calculated from combining a
	number of the old columns with the provided weights. Checks to make sure that
	the weights refer to columns which exist in the passed in dataframe.

	Args:
	    df (pandas.DataFrame): A dataframe with predictor columns to be combined.
	    predictor_columns (list): The names of the predictor columns in a list.
	    weights_dict (dict): Mapping from predictor column names to their weight.
	
	Returns:
	    pandas.DataFrame: The resulting dataframe with columns combined.
	"""
	# Check to make sure the arguments are compatible with each other.
	if not len(set(predictor_columns).difference(set(weights_dict.keys()))) == 0:
		raise Error("Names in the weight dictionary don't match list of predictors.")
	X = _get_X(df, predictor_columns)
	w = np.array([weights_dict[name] for name in predictor_columns])
	W = np.tile(w, (X.shape[0], 1))
	multiplied = np.multiply(X,W)
	y = [np.sum(row) for row in multiplied]
	df = df[["from", "to"]]
	df.loc[:,"similarity"] = y
	return(df)


def apply_mean(df, predictor_columns):
	weight_for_all = 1.000 / float(len(predictor_columns))
	weights_dict = {name:weight_for_all for name in predictor_columns}
	return(apply_weights(df, predictor_columns, weights_dict))










def apply_linear_regression_model(df, predictor_columns, model):
	X = _get_X(df, predictor_columns)
	y = model.predict(X)
	df = df[["from", "to"]]
	df.loc[:,"similarity"] = y
	return(df)




def apply_logistic_regression_model(df, predictor_columns, model, positive_label=1):
	X = _get_X(df, predictor_columns)
	class_probabilities = model.predict_proba(X)
	positive_class_label = positive_label
	positive_class_index = model.classes_.tolist().index(positive_class_label)
	positive_class_probs = [x[positive_class_index] for x in class_probabilities]
	df = df[["from", "to"]]
	df.loc[:,"similarity"] = positive_class_probs
	return(df)



def apply_random_forest_model(df, predictor_columns, model, positive_label=1):
	X = _get_X(df, predictor_columns)
	class_probabilities = model.predict_proba(X)
	positive_class_label = positive_label
	positive_class_index = model.classes_.tolist().index(positive_class_label)
	positive_class_probs = [x[positive_class_index] for x in class_probabilities]
	df = df[["from", "to"]]
	df.loc[:,"similarity"] = positive_class_probs
	return(df)













def train_linear_regression_model(df, predictor_columns, target_column):
	X,y = _get_X_and_y(df, predictor_columns, target_column) 
	reg = LinearRegression().fit(X, y)
	training_rmse = reg.score(X,y)
	coefficients = reg.coef_
	intercept = reg.intercept_ 
	return(reg)



def train_logistic_regression_model(df, predictor_columns, target_column, solver="liblinear", seed=None):
	X,y = _get_X_and_y(df, predictor_columns, target_column)
	lrg = LogisticRegression(random_state=seed, solver=solver)
	lrg.fit(X,y)
	return(lrg)



def train_random_forest_model(df, predictor_columns, target_column, num_trees=100, function="gini", max_depth=None, seed=None):
	X,y = _get_X_and_y(df, predictor_columns, target_column)
	rf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, criterion=function, random_state=seed)
	rf.fit(X,y)
	feature_importance = rf.feature_importances_
	return(rf)










def _get_X_and_y(df, predictor_columns, target_column):
	"""Get arrays for X and y from the dataframe.
	"""
	feature_sets = df[predictor_columns].values.tolist()
	target_values = df[target_column].values.tolist()
	X = np.array(feature_sets)
	y = np.array(target_values)
	return(X,y)



def _get_X(df, predictor_columns):
	"""Get array for X from the dataframe.
	"""
	feature_sets = df[predictor_columns].values.tolist()
	X = np.array(feature_sets)
	return(X)



















