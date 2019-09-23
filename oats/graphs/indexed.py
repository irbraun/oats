import numpy as np
import pandas as pd




class IndexedGraph:



	def __init__(self, df, value):

		self.ids_in_graph = self._get_ids_in_graph(df)
		self.indexed_df = self._index_df(df, value)




	def get_ids_in_graph(self):
		return(self.ids_in_graph)



	def get_value(self, id1, id2):
		"""Summary
		Args:
		    id1 (TYPE): Description
		    id2 (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		try:
			return(self.indexed_df.loc[(id1,id2)]["value"])
		except(KeyError, TypeError):
			pass
		try: 
			return(self.indexed_df.loc[(id2,id1)]["value"])
		except(KeyError, TypeError):
			raise KeyError("no value for this pair of IDs")






	def _get_ids_in_graph(self, df):
		ids_set = set()
		ids_set.update(list(pd.unique(df["from"].values)))
		ids_set.update(list(pd.unique(df["to"].values)))
		ids = list(ids_set)
		return(ids)



	def _index_df(self, df, value):
		indexed_df = df[["from","to",value]]
		indexed_df.columns = ["from", "to", "value"]
		indexed_df.set_index(["from","to"], inplace=True)
		return(indexed_df)


